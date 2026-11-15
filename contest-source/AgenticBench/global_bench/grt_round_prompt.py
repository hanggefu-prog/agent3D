"""Host-only prompt preparation, with no model calls, budget spending or evaluation."""
import argparse
import copy
import json
from pathlib import Path

from grt_bench.common import digest, finite, read, records, write
from grt_bench.contract import timing_only, validate
from grt_bench.optimization import identity, prompt_scope
from grt_bench.pareto import aggregate, frontier
from .grt_candidate_library import check_summaries, load_library, visible_path

TEMPLATES = Path(__file__).resolve().parents[1] / "global_tasks/lol_post_grt_pareto"


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def text_schema():
    return {"type": "string"}


def array_schema(item):
    return {"type": "array", "items": item}


def _check(value, schema):
    """Check only the small schema vocabulary emitted by this module."""
    kinds = {"object": dict, "array": list, "string": str, "integer": int, "null": type(None)}
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    if not any(type(value) is kinds[k] for k in types):
        raise ValueError("Response field has incorrect type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Response field outside frozen choices")
    if isinstance(value, dict):
        if set(value) != set(schema["properties"]):
            raise ValueError("Missing/unknown response fields")
        for k, v in value.items():
            _check(v, schema["properties"][k])
    elif isinstance(value, list):
        for v in value:
            _check(v, schema["items"])


def check_parent_request(bundle, request):
    if bundle.get("schema_version") != 3 or bundle.get("phase") != "round":
        raise ValueError("Requires a single-invocation round prompt")
    _check(request, bundle["parent_tool"]["inputSchema"])
    if not request["reason"].strip() or not request["hypothesis"].strip() or not request["files_to_modify"]:
        raise ValueError("Parent selection requires a reason, hypothesis and files")
    if not request["evidence"] or any(not s.strip() for s in request["evidence"]):
        raise ValueError("Parent selection requires specific supplied evidence")
    if len(set(request["files_to_modify"])) != len(request["files_to_modify"]):
        raise ValueError("Duplicate planned files")
    return request


def check_response(bundle, response, checkout=None):
    if bundle.get("schema_version") != 3 or bundle.get("phase") != "round":
        raise ValueError("Requires a single-invocation round prompt")
    _check(response, bundle["output_schema"])
    files = response["changed_files"]
    check_summaries(response["change_summary"], required=response["status"] == "ready")
    if not response["summary"].strip():
        raise ValueError("Final response requires a summary or blocking reason")
    if checkout is None:
        if (response["status"] != "blocked" or response["parent"] is not None
                or response["selection_id"] is not None or files or response["files_to_modify"]
                or any(response["change_summary"].values())):
            raise ValueError("No host checkout receipt; only a pre-checkout blocked result is valid")
    else:
        context = bundle["context"]
        if (checkout["bundle_digest"] != digest(bundle) or checkout["task_digest"] != context["task_digest"]
                or checkout["round"] != context["budget"]["round_number"]
                or checkout["workspace"] != context["workspace"]
                or checkout["parent"] != response["parent"]
                or checkout["selection_id"] != response["selection_id"]):
            raise ValueError("Final response differs from host checkout receipt")
        proposal = {k: response[k] for k in bundle["parent_tool"]["inputSchema"]["properties"]}
        check_parent_request(bundle, proposal)
        # Planning prose may evolve after source inspection. Identity and source
        # scope remain host-bound; actual edits are verified independently.
    if response["status"] == "ready" and not files:
        raise ValueError("Ready requires actual changed files")
    if response["status"] == "ready" and any(c["outcome"] == "failed" for c in response["checks"]):
        raise ValueError("Ready candidate has a failed local check")
    if response["status"] == "no_change" and files:
        raise ValueError("No-change response declares changed files")
    if len(set(files)) != len(files):
        raise ValueError("Duplicate changed files")
    return response


def bind_response(bundle, response, checkout):
    """Canonicalize redundant receipt fields; never accept a different chosen parent.

    Task/round, source permissions and the authenticated checkout are still checked.
    The caller persists the unmodified response and this audit before evaluation.
    """
    if not isinstance(response, dict):
        raise ValueError("Final response must be a JSON object")
    result = copy.deepcopy(response)
    changes = {}
    if checkout is not None:
        # Null or the literal field-guide placeholder carries no identity claim.
        # An actual different digest is a conflicting parent, never a typo to repair.
        parent = result.get("parent")
        if parent not in {None, checkout["parent"], "frontier candidate ID"}:
            raise ValueError("Final response selects a different parent")
        for key in ("parent", "selection_id"):
            if result.get(key) != checkout[key]:
                changes[key] = {"submitted": result.get(key), "host": checkout[key]}
            result[key] = checkout[key]
    check_response(bundle, result, checkout)
    return result, {"receipt_fields_bound": changes,
        "proposal_revised": bool(checkout and any(result[k] != checkout["proposal"][k]
            for k in bundle["parent_tool"]["inputSchema"]["properties"]))}


def _project_row(task, row):
    training = {c["id"] for c in task["cases"] if c["split"] == "train"}
    if (row["task_digest"] != validate(task) or row["split"] != "train"
            or set(row["per_case"]) != training or set(row["vector"]) != set(task["objectives"])
            or type(row.get("feasible")) is not bool
            or set(row["metric_ratios"]) != set(task["metrics"])
            or row["cells"] != len(training) * len(task["seeds"])):
        raise ValueError("Feedback must be a complete training row under this task")
    identity(row["candidate"]); identity(row["reference_digest"])
    per_case = {}
    for case in sorted(training):
        if set(row["per_case"][case]) != set(task["metrics"]):
            raise ValueError("Feedback objective set differs")
        per_case[case] = {}
        for metric in task["metrics"]:
            values = row["per_case"][case][metric]
            if len(values["samples"]) != len(task["seeds"]):
                raise ValueError("Feedback seed count differs")
            per_case[case][metric] = {k: finite(values[k], k, 0) for k in ("mean", "stdev", "worst")}
            per_case[case][metric]["samples"] = [finite(v, "sample", 0) for v in values["samples"]]
    return {"candidate": row["candidate"], "vector": {m: finite(row["vector"][m], m, 0) for m in task["objectives"]},
            "metric_ratios": {m: finite(row["metric_ratios"][m], m, 0) for m in task["metrics"]},
            "feasible": row["feasible"], "constraints": copy.deepcopy(row["constraints"]),
            "per_case_ratios": per_case}


def _agent_context(context):
    """Keep decisions in the prompt; retain full precision and host audit data."""
    view = {k: copy.deepcopy(context[k]) for k in (
        "task_digest", "workspace", "scratch_directory", "cells_per_matrix", "components", "baseline")}
    view["budget"] = {k: v for k, v in context["budget"].items() if k != "round_reserved"}
    view["scoring"] = copy.deepcopy(context["contract"]["scoring"])
    view["normalization_floors"] = {k: v for k, v in context["contract"]["normalization_floors"].items()
                                    if k != "power_w"}  # Power uses raw watts, not this floor.
    view["candidate_index"] = context["candidate_library"]["index"]
    if "task_details" in context:
        view["task_details"] = context["task_details"]
    else:
        # Compatibility with sealed libraries from before the details export.
        view["allowed_files"] = copy.deepcopy(context["allowed_files"])
    view["candidates"] = []
    for entry in context["candidates"]:
        item = {k: copy.deepcopy(entry[k]) for k in
                ("candidate", "parent", "status", "change_summary", "evaluation_summary", "directory")}
        if item["evaluation_summary"] is not None:
            for key in ("delta_from_parent", "power_failed_cases"):
                if key in item["evaluation_summary"] and not item["evaluation_summary"][key]:
                    del item["evaluation_summary"][key]
        view["candidates"].append(item)
    if context["recent_history"]:
        view["recent_history"] = [{k: v for k, v in item.items() if k != "split"}
                                  for item in context["recent_history"]]
    if context["omitted_history_count"]:
        view["omitted_history_count"] = context["omitted_history_count"]
    if "local_check_limits" in context:
        view["local_check_limits"] = copy.deepcopy(context["local_check_limits"])
    return view


def _field_guide(schema, field=""):
    """Display field shapes once, without copying long scope/identity enums.

    The original schemas still bind the MCP tool, structured output and checks.
    This is a reading guide, not a replacement validation schema.
    """
    if field == "parent":
        return "frontier candidate ID" + (" or null" if "null" in schema["type"] else "")
    if field == "task_digest":
        return "HOST_CONTEXT.task_digest"
    if field in {"files_to_modify", "changed_files"}:
        return ["path in allowed_files (read task_details, or the inline list)"]
    if schema["type"] == "object":
        return {k: _field_guide(v, k) for k, v in schema["properties"].items()}
    if schema["type"] == "array":
        return [_field_guide(schema["items"])]
    if "enum" in schema:
        return " | ".join(v if isinstance(v, str) else json.dumps(v) for v in schema["enum"])
    return " or ".join(schema["type"]) if isinstance(schema["type"], list) else schema["type"]


def _bundle(context, schema, parent_schema, background=None):
    templates = TEMPLATES.with_name("lol_post_grt_timing") if timing_only(context["contract"]) else TEMPLATES
    sources = {name: (templates / name).read_text() for name in ("background.md", "round.md")}
    # Fresh sessions need the task background even after the first round.
    if background is None:
        background = context["budget"]["round_number"] == 1
    sections = ([sources["background.md"]] if background else [])
    sections.append(sources["round.md"])
    cases = ["## Evaluation cases and seeds", "",
             "This run evaluates every case below with every seed row. You may also use",
             "these cases for local tests.", "",
             "| Case | Design | PDK | Weight |", "| --- | --- | --- | --- |"]
    cases.extend("| {id} | {design} | {pdk} | {weight} |".format(**case)
                 for case in context["training_cases"])
    cases.extend(["", "| Seed | Partition | Placement |",
                  "| --- | --- | --- |"])
    cases.extend("| {id} | {partition} | {placement} |".format(**seed)
                 for seed in context["contract"]["seeds"])
    cases.extend(["", "The I/O seed stays fixed at 1.",
                  "HBT, OPT and CTS have no separately varied seed in this task."])
    sections.append("\n".join(cases))
    tool = {"name": "checkout_parent", "description": "Record one parent and prepare its bound workspace for this round; no evaluation.",
            "inputSchema": parent_schema}
    agent_context = _agent_context(context)
    prompt = "\n\n".join(sections) + "\n\nHOST_CONTEXT (experiment harness data):\n" + json.dumps(agent_context, indent=2, sort_keys=True)
    prompt += ("\n\nField guides below describe types and choices, not literal response values. "
               "All fields are required; arrays may be empty where the rules allow. "
               "The tool and structured output carry the full validation schemas.\n")
    prompt += "\nPARENT_TOOL: checkout_parent\n" + json.dumps(_field_guide(parent_schema), indent=2)
    prompt += "\n\nOUTPUT_FORMAT:\n" + json.dumps(_field_guide(schema), indent=2) + "\n"
    return {"schema_version": 3, "phase": "round", "context": context, "output_schema": schema,
            "agent_context": agent_context, "parent_tool": tool, "prompt": prompt, "template_digest": digest(sources)}


def round_prompt(task, archive_view, budget, outcomes, history=(), *, library,
                 workspace="/workspace/subject", scratch_directory="/workspace/checks", local_check_limits=None,
                 research_brief=None, background=None):
    if background is not None and type(background) is not bool:
        raise ValueError("background must be a JSON boolean")
    task = copy.deepcopy(task)
    workspace_path = visible_path(workspace)
    scratch_path = visible_path(scratch_directory)
    task_id = validate(task)
    if "optimization" not in task or archive_view["task_digest"] != task_id:
        raise ValueError("Requires the matching frozen optimization archive")
    frozen_brief = task["optimization"]["prompt"]
    if research_brief is None:
        research_brief = frozen_brief
    # A presentation revision cannot change source permissions or evaluator identity.
    scope = lambda text: {k: sorted(v) for k, v in prompt_scope(text).items()}
    if not isinstance(research_brief, str) or scope(research_brief) != scope(frozen_brief):
        raise ValueError("Revised research brief must preserve the frozen source scope")
    if set(budget) != {"round_number", "round_limit", "evaluation_limit", "evaluations_used", "round_reserved"}:
        raise ValueError("Missing/unknown host budget fields")
    for k in ("round_number", "round_limit", "evaluation_limit", "evaluations_used"):
        if type(budget[k]) is not int:
            raise ValueError("Budget counters must be integers")
    if (budget["round_reserved"] is not True or not 1 <= budget["round_number"] <= budget["round_limit"]
            or not 0 <= budget["evaluations_used"] < budget["evaluation_limit"] <= budget["round_limit"]
            or budget["evaluations_used"] >= budget["round_number"]):
        raise ValueError("No reserved round/evaluation capacity remains")
    rows = archive_view["evaluated"]
    projected = {row["candidate"]: _project_row(task, row) for row in rows}
    baseline = task["optimization"]["baseline"]
    if baseline not in projected or len(projected) != len(rows):
        raise ValueError("A unique qualified baseline and history are required")
    current = frontier(rows, retain_equivalent=True)
    if archive_view["frontier"] != current:
        raise ValueError("Training frontier differs from archived evaluations")
    by_candidate = {}
    for outcome in outcomes:
        if outcome.get("split") != "train" or outcome.get("task_digest") != task_id:
            raise ValueError("Raw feedback must contain only this task's training results")
        by_candidate.setdefault(outcome["candidate"], []).append(outcome)
    required = {baseline, *(r["candidate"] for r in current)}
    if set(by_candidate) != required:
        raise ValueError("Supply raw results for exactly the baseline and current frontier")
    archived = {r["candidate"]: r for r in rows}
    for candidate in sorted(required):
        raw = by_candidate[candidate]
        if aggregate(task, candidate, raw, by_candidate[baseline]) != archived[candidate]:
            raise ValueError("Raw results differ from admitted archive evidence")
    library_index, library_digest = load_library(library, task, archive_view)
    library_path = Path(library_index["agent_root"])
    if (workspace_path == library_path or workspace_path in library_path.parents
            or library_path in workspace_path.parents):
        raise ValueError("Reading library and editable workspace must be separate")
    if any(scratch_path == path or scratch_path in path.parents or path in scratch_path.parents
           for path in (workspace_path, library_path)):
        raise ValueError("Scratch directory must be separate from source and reading library")
    catalog = []
    for entry in library_index["candidates"]:
        catalog.append({**{k: entry[k] for k in ("candidate", "parent", "status", "selectable", "evaluation_summary")},
            "change_summary": {k: v["summary"] for k, v in entry["changes"].items()},
            "summary_origin": {k: v["summary_origin"] for k, v in entry["changes"].items()},
            "directory": entry["paths"]["directory"]})
    prior = []
    for item in history:
        if set(item) != {"round", "split", "status", "parent", "candidate", "hypothesis", "feedback"}:
            raise ValueError("Unexpected history fields; export a training-only view")
        if item["split"] != "train" or type(item["round"]) is not int or not 1 <= item["round"] < budget["round_number"]:
            raise ValueError("Held-out/current/future round feedback is forbidden")
        if item["status"] not in {"verified", "infeasible", "rejected", "failed", "no_change", "blocked", "generation_failed"}:
            raise ValueError("Unknown round status")
        for k in ("parent", "candidate"):
            if item[k] is not None: identity(item[k])
        for k in ("hypothesis", "feedback"):
            if not isinstance(item[k], str): raise ValueError("History explanations must be text")
        prior.append(copy.deepcopy(item))
    if len({p["round"] for p in prior}) != len(prior):
        raise ValueError("Duplicate history round")
    prior.sort(key=lambda p: p["round"])
    context = {"task_digest": task_id, "budget": copy.deepcopy(budget), "workspace": workspace,
        "scratch_directory": str(scratch_path),
        "maximum_model_invocations_this_round": 1,
        "direct_full_evaluation_allowance": 0, "maximum_matrices_this_round": 1,
        "cells_per_matrix": len(task["seeds"]) * sum(c["split"] == "train" for c in task["cases"]),
        "contract": {k: copy.deepcopy(task[k]) for k in ("endpoint", "metrics", "objectives", "scoring", "normalization_floors", "opt_passes", "activity", "ambient_k", "grid_size", "seeds") if k in task},
        "training_cases": [{k: c[k] for k in ("id", "design", "pdk", "weight")} for c in task["cases"] if c["split"] == "train"],
        "components": task["optimization"]["components"],
        "allowed_files": task["optimization"]["allowed_files"], "baseline": baseline,
        "frontier": [r["candidate"] for r in current], "candidates": catalog,
        "candidate_library": {"directory": str(library_path), "index": str(library_path / "index.json"),
                              "manifest_sha256": library_digest},
        "reference_digest": rows[0]["reference_digest"], "recent_history": prior[-8:],
        "omitted_history_count": max(0, len(prior) - 8)}
    if (Path(library) / "task-context.json").exists():
        context["task_details"] = str(library_path / "task-context.json")
    if local_check_limits is not None:
        expected = {"cpu", "memory_gb", "gpu_count", "round_seconds", "tool_seconds", "source_write_mode"}
        if set(local_check_limits) - {'gpu_check'} != expected:
            raise ValueError("Missing/unknown local-check limits")
        for key in ("cpu", "memory_gb", "round_seconds", "tool_seconds"):
            if type(local_check_limits[key]) is not int or local_check_limits[key] < 1:
                raise ValueError("Local-check limits must be positive integers")
        if local_check_limits["gpu_count"] != 0 or local_check_limits["source_write_mode"] not in {"in-place", "development-copy"}:
            raise ValueError("Unsupported development environment mode")
        if 'gpu_check' in local_check_limits:
            from .grt_gpu_checks import limits
            value = dict(local_check_limits['gpu_check'])
            if value.pop('gpu_count', None) != 1:
                raise ValueError('GPU checks require one scheduled GPU')
            limits({'gpu_checks': value})
        context["local_check_limits"] = copy.deepcopy(local_check_limits)
    string = text_schema()
    parents = [r["candidate"] for r in current]
    proposal = {"parent": {"type": "string", "enum": parents},
        "reason": string, "hypothesis": string, "evidence": array_schema(string),
        "files_to_modify": array_schema({"type": "string", "enum": context["allowed_files"]}),
        "expected_tradeoffs": obj({m: {"type": "string", "enum": ["improve", "regress", "unchanged", "uncertain"]} for m in (("timing", "overflow_penalty") if timing_only(task) else (*task["objectives"], "power_w"))})}
    schema = obj({**proposal, "parent": {"type": ["string", "null"], "enum": [None] + parents},
        "task_digest": {"type": "string", "enum": [task_id]},
        "round": {"type": "integer", "enum": [budget["round_number"]]},
        "selection_id": {"type": ["string", "null"]},
        "status": {"type": "string", "enum": ["ready", "no_change", "blocked"]},
        "summary": string, "change_summary": obj({k: string for k in ("from_reference", "from_parent")}),
        "changed_files": array_schema({"type": "string", "enum": context["allowed_files"]}),
        "checks": array_schema(obj({"name": string, "outcome": {"type": "string", "enum": ["passed", "failed", "not_run"]}, "details": string})),
        "risks": array_schema(string), "next_round_hint": string})
    bundle = _bundle(context, schema, obj(proposal), background)
    # The task brief is host audit material. background.md and round.md own the
    # agent-facing prose; embedding the brief in HOST_CONTEXT duplicates it.
    bundle["research_brief_provenance"] = {"frozen_digest": digest(frozen_brief),
                                         "audit_digest": digest(research_brief)}
    return bundle


def save_bundle(output, bundle):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "bundle.json", bundle)
    write(output / "output.schema.json", bundle["output_schema"])
    write(output / "parent-tool.json", bundle["parent_tool"])
    (output / "prompt.md").write_text(bundle["prompt"])
    write(output / "manifest.json", {"bundle_digest": digest(bundle), "artifacts": records(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("task", "archive", "budget", "outcomes", "output"): parser.add_argument(key, type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--workspace", default="/workspace/subject")
    parser.add_argument("--scratch-directory", default="/workspace/checks")
    parser.add_argument("--research-brief", type=Path,
                        help="Reviewed presentation revision with unchanged source-scope front matter")
    args = parser.parse_args()
    bundle = round_prompt(read(args.task), read(args.archive), read(args.budget),
                          read(args.outcomes), read(args.history) if args.history else [],
                          workspace=args.workspace, scratch_directory=args.scratch_directory, library=args.library,
                          research_brief=args.research_brief.read_text() if args.research_brief else None)
    save_bundle(args.output, bundle)


if __name__ == "__main__":
    main()
