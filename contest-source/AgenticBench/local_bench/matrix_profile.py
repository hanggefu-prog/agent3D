"""Declarative default evaluation-matrix profile for the maintained local tasks.

A profile names the cases, paired seed vectors and per-task settings of a
matrix. :func:`render_matrix` turns it plus one frozen upstream baseline run
per cell into the existing schema-1 ``matrix.json`` read by
:class:`local_bench.matrix.LocalEvaluationMatrix`. Rendering does not run any
EDA tool and does not create references: the result must still be qualified
with ``qualify-matrix``. The single-case task interfaces are unchanged.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .model import load_task
from .seeds import validate_schedule
from .task_inventory import MATRIX_TASK_DIRECTORIES


DEFAULT_MATRIX_PROFILE = Path(__file__).resolve().parent / "reference_configs/default_matrix_profile.yaml"
DEFAULT_TASK_ROOT = Path(__file__).resolve().parents[1] / "local_tasks"
PROFILE_KEYS = {
    "schema_version", "id", "description", "source_release", "qualified",
    "timeout_seconds", "cases", "seeds", "tasks",
}
TASK_KEYS = {"workers", "base_config", "config_overrides", "case_config_overrides"}
SEED_STAGES = (("pins", "io"), ("partition", "partition"),
               ("placement", "placement"), ("routing", "routing"))
# Matrix identity and paired seeds are owned by the profile, never by overrides.
PROTECTED_TOP_LEVEL = {"design", "type", "pdk"}


def _check_override_mapping(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping of dotted config paths".format(label))
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not all(key.split(".")):
            raise ValueError("{} has an invalid config path: {!r}".format(label, key))
        parts = key.split(".")
        if parts[0] in PROTECTED_TOP_LEVEL or parts[-1] == "seed":
            raise ValueError(
                "{} may not override {}; cases and seeds own design/platform and seeds".format(label, key)
            )
        if isinstance(item, (Mapping, list)):
            raise ValueError("{} value for {} must be a scalar".format(label, key))
        result[key] = item
    return result


def load_matrix_profile(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate one profile; return a normalized dictionary."""

    profile_path = Path(path or DEFAULT_MATRIX_PROFILE).expanduser().resolve(strict=True)
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != PROFILE_KEYS:
        raise ValueError(
            "{}: matrix profile requires exactly the keys {}".format(
                profile_path, ", ".join(sorted(PROFILE_KEYS)))
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("{}: unsupported matrix profile schema_version".format(profile_path))
    if data["qualified"] is not False:
        # Qualification is evidence produced by qualify-matrix on frozen inputs;
        # a static profile can never claim it.
        raise ValueError("{}: a matrix profile must declare qualified: false".format(profile_path))
    timeout = data["timeout_seconds"]
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("{}: timeout_seconds must be a positive integer".format(profile_path))
    cases = data["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("{}: cases must be a nonempty list".format(profile_path))
    case_ids = []
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"id", "design"}
                or not all(isinstance(case[k], str) and case[k] for k in ("id", "design"))):
            raise ValueError("{}: each case needs a string id and design".format(profile_path))
        case_ids.append(case["id"])
    if len(set(case_ids)) != len(case_ids) or len({c["design"] for c in cases}) != len(cases):
        raise ValueError("{}: duplicate case id or design".format(profile_path))
    seeds = data["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("{}: seeds must be a nonempty list".format(profile_path))
    seed_ids, vectors = [], []
    for seed in seeds:
        if not isinstance(seed, dict) or not isinstance(seed.get("id"), str) or not seed["id"]:
            raise ValueError("{}: each seed vector needs a string id".format(profile_path))
        seed_ids.append(seed["id"])
        vectors.append(json.dumps(validate_schedule(
            {k: v for k, v in seed.items() if k != "id"}), sort_keys=True))
    if len(set(seed_ids)) != len(seed_ids) or len(set(vectors)) != len(vectors):
        raise ValueError("{}: duplicate seed id or seed vector".format(profile_path))
    tasks = data["tasks"]
    if not isinstance(tasks, dict) or set(tasks) != set(MATRIX_TASK_DIRECTORIES):
        raise ValueError(
            "{}: tasks must list exactly the maintained task directories with a seed-matrix contract {}".format(
                profile_path, ", ".join(MATRIX_TASK_DIRECTORIES))
        )
    normalized_tasks = {}
    for name in MATRIX_TASK_DIRECTORIES:
        entry = tasks[name]
        label = "{}: tasks.{}".format(profile_path, name)
        if not isinstance(entry, dict) or not set(entry) <= TASK_KEYS or "workers" not in entry:
            raise ValueError("{} needs workers and only {}".format(label, ", ".join(sorted(TASK_KEYS))))
        if type(entry["workers"]) is not int or entry["workers"] < 1:
            raise ValueError("{}.workers must be a positive integer".format(label))
        base_config = None
        if entry.get("base_config") is not None:
            base_config = (profile_path.parent / str(entry["base_config"])).resolve()
            if not base_config.is_file():
                raise ValueError("{}.base_config does not exist: {}".format(label, base_config))
        per_case = entry.get("case_config_overrides") or {}
        if not isinstance(per_case, Mapping) or not set(per_case) <= set(case_ids):
            raise ValueError("{}.case_config_overrides must be keyed by case id".format(label))
        normalized_tasks[name] = {
            "workers": entry["workers"],
            "base_config": base_config,
            "config_overrides": _check_override_mapping(
                entry.get("config_overrides"), label + ".config_overrides"),
            "case_config_overrides": {
                case: _check_override_mapping(value, "{}.case_config_overrides.{}".format(label, case))
                for case, value in per_case.items()
            },
        }
    result = deepcopy(data)
    result["tasks"] = normalized_tasks
    result["path"] = profile_path
    return result


def _set_dotted(config: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target = config
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError("config path {} crosses a non-mapping value".format(dotted))
        target = child
    target[parts[-1]] = value


def cell_config(profile: Mapping[str, Any], task_directory: str, case_id: str, seed_id: str,
                *, task_root: Optional[Path] = None) -> Dict[str, Any]:
    """Return the resolved base configuration for one matrix cell."""

    if task_directory not in profile["tasks"]:
        raise ValueError("matrix profile has no entry for task {}".format(task_directory))
    entry = profile["tasks"][task_directory]
    spec = load_task(Path(task_root or DEFAULT_TASK_ROOT) / task_directory)
    source = entry["base_config"] or spec.base_config
    config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("base configuration must be a mapping: {}".format(source))
    case = {c["id"]: c for c in profile["cases"]}[case_id]
    seed = {s["id"]: s for s in profile["seeds"]}[seed_id]
    config["design"] = case["design"]
    for dotted, value in entry["config_overrides"].items():
        _set_dotted(config, dotted, value)
    for dotted, value in entry["case_config_overrides"].get(case_id, {}).items():
        _set_dotted(config, dotted, value)
    for stage, key in SEED_STAGES:
        stage_config = config.setdefault(stage, {})
        if not isinstance(stage_config, dict):
            raise ValueError("config stage {} must be a mapping".format(stage))
        stage_config["seed"] = seed[key]
    return config


def render_matrix(task_directory: str, baseline_runs: Mapping[str, Mapping[str, Any]],
                  output_dir: Path, *, profile: Optional[Mapping[str, Any]] = None,
                  task_root: Optional[Path] = None) -> Path:
    """Write per-cell configs and an unqualified schema-1 ``matrix.json``.

    ``baseline_runs`` maps ``case id -> seed id -> frozen upstream run`` and
    must cover every cell. ``reference_run`` is left ``null``; run
    ``qualify-matrix`` on the result to create qualified references.
    """

    profile = profile or load_matrix_profile()
    if task_directory not in MATRIX_TASK_DIRECTORIES:
        raise ValueError("Task has no defined seed-matrix contract: " + task_directory)
    spec = load_task(Path(task_root or DEFAULT_TASK_ROOT) / task_directory)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells = []
    for case in profile["cases"]:
        for seed in profile["seeds"]:
            try:
                baseline = baseline_runs[case["id"]][seed["id"]]
            except (KeyError, TypeError):
                raise ValueError("missing baseline run for cell {}/{}".format(case["id"], seed["id"]))
            if not isinstance(baseline, (str, Path)) or not str(baseline):
                raise ValueError("baseline run for {}/{} must be a path".format(case["id"], seed["id"]))
            config = cell_config(profile, task_directory, case["id"], seed["id"], task_root=task_root)
            path = output / "configs" / case["id"] / (seed["id"] + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            cells.append({
                "case": case["id"],
                "seed": seed["id"],
                "base_config": str(path),
                "baseline_run": str(Path(baseline).expanduser().resolve()),
                "reference_run": None,
            })
    matrix = {
        "schema_version": 1,
        "task": spec.task_id,
        "cases": [case["id"] for case in profile["cases"]],
        "seeds": deepcopy(profile["seeds"]),
        "cells": cells,
    }
    matrix_path = output / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    return matrix_path


def task_settings(task_directory: str, profile: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the qualification/search settings the profile fixes for one task."""

    profile = profile or load_matrix_profile()
    entry = profile["tasks"][task_directory]
    return {"workers": entry["workers"], "timeout_seconds": profile["timeout_seconds"]}


__all__ = [
    "DEFAULT_MATRIX_PROFILE", "cell_config", "load_matrix_profile",
    "render_matrix", "task_settings",
]
