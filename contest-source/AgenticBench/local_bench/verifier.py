from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .model import TaskSpec, validate_candidate_payload
from .policy import flatten_paths, validate_config_scope, validate_patch_text


def _pointer(value: Any, path: str) -> Any:
    if not path.startswith("/"):
        raise ValueError("constraint path must be a JSON pointer: {}".format(path))
    cursor = value
    for token in path[1:].split("/") if path != "/" else []:
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(cursor, Mapping) or token not in cursor:
            raise KeyError(path)
        cursor = cursor[token]
    return cursor


def _check_type(path: str, value: Any, expected: str) -> None:
    valid = {
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, Mapping),
    }.get(expected)
    if valid is None:
        raise ValueError("unsupported constraint type {!r} for {}".format(expected, path))
    if not valid:
        raise ValueError("candidate value {} must have type {}".format(path, expected))
    if expected in {"integer", "number"} and not math.isfinite(float(value)):
        raise ValueError("candidate value {} must be finite".format(path))


def validate_value_constraints(spec: TaskSpec, candidate: Mapping[str, Any]) -> None:
    for rule in spec.value_constraints:
        path = str(rule.get("path", ""))
        try:
            value = _pointer(candidate, path)
        except KeyError:
            if rule.get("required", True):
                raise ValueError("candidate is missing required value {}".format(path))
            continue
        expected = rule.get("type")
        if expected:
            _check_type(path, value, str(expected))
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError("candidate value {} is not in {}".format(path, rule["enum"]))
        if "minimum" in rule and value < rule["minimum"]:
            raise ValueError("candidate value {} is below {}".format(path, rule["minimum"]))
        if "maximum" in rule and value > rule["maximum"]:
            raise ValueError("candidate value {} is above {}".format(path, rule["maximum"]))
        if "exclusive_minimum" in rule and value <= rule["exclusive_minimum"]:
            raise ValueError("candidate value {} must exceed {}".format(path, rule["exclusive_minimum"]))
        if "exclusive_maximum" in rule and value >= rule["exclusive_maximum"]:
            raise ValueError("candidate value {} must be below {}".format(path, rule["exclusive_maximum"]))


def verify_candidate(
    spec: TaskSpec, value: Any, open3dbench_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Validate one candidate against the task contract.

    Any evaluator-wrapper hunk in a
    routing source candidate must be byte-identical to the trusted task adapter
    output re-derived from that checkout (see ``routing_wrapper_integrity``).
    Wrapper hunks require ``open3dbench_root``; pure source-only scope checks
    may omit it. Every evaluation entry point supplies the selected checkout.
    """
    candidate = validate_candidate_payload(value)
    validate_config_scope(spec, candidate.get("config_overrides", {}))
    declared = {item.path for item in spec.candidate_files}
    embedded = candidate.get("candidate_files", {})
    unknown = sorted(set(embedded) - declared)
    if unknown:
        raise ValueError("candidate contains undeclared files: {}".format(", ".join(unknown)))
    for item in spec.candidate_files:
        if item.path not in embedded or not item.allowed_paths:
            continue
        changed = flatten_paths(embedded[item.path])
        rejected = sorted(
            path
            for path in changed
            if not any(
                path == allowed or allowed.startswith(path + ".")
                for allowed in item.allowed_paths
            )
        )
        if rejected:
            raise ValueError(
                "candidate file {} contains unsupported fields: {}".format(
                    item.path, ", ".join(rejected)
                )
            )
    patch = validate_patch_text(spec, candidate.get("source_patch", ""))
    from .routing_wrapper_integrity import verify_wrapper_hunks
    verify_wrapper_hunks(spec, candidate.get("source_patch", ""), open3dbench_root)
    validate_value_constraints(spec, candidate)
    if spec.kind == "hbt_assignment_policy":
        from .hbt_policy import validate_candidate
        validate_candidate(candidate)
    from local_tasks.tp_gnn_local_task_definitions import TASKS as PARTITION_DEFINITIONS
    if spec.kind in PARTITION_DEFINITIONS:
        from local_tasks.tp_gnn_local_task_definitions import validate_candidate as validate_partition_program
        validate_partition_program(candidate, spec.kind)
    if spec.kind in {"placement_parameters", "placement_dynamic_parameters"}:
        public = candidate.get("config_overrides", {}).get("placement", {}).get("target_density")
        private = embedded.get("placement.json", {}).get("concurrent_dreamplace", {}).get("target_density")
        if public is None or private is None or abs(float(public) - float(private)) > 1.0e-12:
            raise ValueError(
                "placement.target_density and concurrent_dreamplace.target_density must match"
            )
    return {"candidate": candidate, "patch": patch}
