from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .routing_scoring import parse_scoring


ROUTING_SOURCE_TASK_KINDS = frozenset(
    {
        "routing_net_ordering_source",
        "routing_ripup_policy_source",
        "routing_maze_cost_source",
        "routing_iteration_controller_source",
    }
)


@dataclass(frozen=True)
class CandidateFile:
    path: str
    inject_as: str
    allowed_paths: List[str]


@dataclass(frozen=True)
class SourceRange:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class FrozenInputJsonCheck:
    path: str
    equals: Dict[str, Any]


@dataclass(frozen=True)
class FrozenInput:
    """One immutable evaluator-owned input adopted from a completed run."""

    name: str
    source: str
    path: str
    inject_as: str
    required_files: List[str]
    json_checks: List[FrozenInputJsonCheck] = field(default_factory=list)


@dataclass(frozen=True)
class TaskSpec:
    root: Path
    task_id: str
    title: str
    description: str
    kind: str
    base_config: Path
    initial_candidate: Path
    prompt: Path
    start_stage: str
    stop_stage: str
    upstream_stages: List[str]
    evaluation_boundary: str
    allowed_config_paths: List[str]
    allowed_source_paths: List[str]
    candidate_files: List[CandidateFile]
    value_constraints: List[Dict[str, Any]]
    max_changed_files: int = 0
    max_added_lines: int = 0
    max_deleted_lines: int = 0
    prompt_source_ranges: List[SourceRange] = field(default_factory=list)
    prompt_read_only_source_paths: List[str] = field(default_factory=list)
    priority_parameters: List[str] = field(default_factory=list)
    max_source_chars: int = 80_000
    feedback_recent: int = 3
    feedback_top: int = 3
    max_feedback_chars: int = 24_000
    program_mode: str = "candidate_literal"
    program_template: Optional[Path] = None
    program_adapter: Optional[Path] = None
    program_entrypoint: str = "evolve_candidate"
    benchmark_input_id: Optional[str] = None
    frozen_inputs: List[FrozenInput] = field(default_factory=list)
    availability: str = "available"
    # Normalized routing score contract (see ``routing_scoring``); empty means
    # the historical version-1 formula and keeps the contract byte-identical.
    scoring: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": self.metrics, "artifacts": self.artifacts}


def _required(data: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise ValueError("{} is missing required key {!r}".format(path, key))
    return data[key]


def resolve_task_path(task_root: Path, value: str) -> Path:
    """Resolve an explicit spec path, directory name, or public spec id.

    Optimization tasks use ``task.yaml``.  Evaluation-only profiles use
    ``profile.yaml`` and are kept outside ``local_tasks`` so they cannot be
    mistaken for search problems.
    """
    def definition(path: Path) -> Optional[Path]:
        for name in ("task.yaml", "profile.yaml"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        return None

    explicit = Path(value).expanduser()
    if explicit.exists():
        return explicit.resolve()

    by_directory = (task_root / value).resolve()
    if definition(by_directory) is not None:
        return by_directory

    matches = []
    for path in sorted(task_root.iterdir()):
        config_path = definition(path)
        if config_path is None:
            continue
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and str(data.get("id", "")) == value:
            matches.append(path.resolve())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("task id {!r} is ambiguous under {}".format(value, task_root))
    raise FileNotFoundError("unknown task path, directory, or id {!r}".format(value))


def load_task(task_dir: Path) -> TaskSpec:
    root = task_dir.expanduser().resolve()
    task_path = root / "task.yaml"
    profile_path = root / "profile.yaml"
    if task_path.is_file() and profile_path.is_file():
        raise ValueError("spec directory cannot contain both task.yaml and profile.yaml: {}".format(root))
    config_path = task_path if task_path.is_file() else profile_path
    if not config_path.is_file():
        raise FileNotFoundError("missing task.yaml or profile.yaml under {}".format(root))
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("task.yaml must contain a mapping: {}".format(config_path))
    if data.get("kind") in {"tp_gnn_ta3d_learning_objective", "tp_gnn_pin_cut_learning_objective"}:
        raise ValueError("Retired cut-score task kind; use tp_gnn_post_placement_timing")
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported task schema_version in {}".format(config_path))
    availability = str(data.get("availability", "available"))
    if availability not in {"available", "legacy", "definition-only"}:
        raise ValueError("unsupported task availability: " + availability)

    execution = _required(data, "execution", config_path)
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object: {}".format(config_path))
    evaluation_boundary = str(execution.get("boundary", "open3dbench-stage"))
    if evaluation_boundary not in {
        "open3dbench-stage",
        "dreamplace-global-placement",
    }:
        raise ValueError(
            "unsupported execution.boundary {!r}: {}".format(
                evaluation_boundary, config_path
            )
        )
    policy = data.get("candidate_policy", {})
    prompt_context = data.get("prompt_context", {})
    if not isinstance(prompt_context, dict):
        raise ValueError("prompt_context must be an object: {}".format(config_path))
    program = data.get("program", {})
    if not isinstance(program, dict):
        raise ValueError("program must be an object: {}".format(config_path))
    program_mode = str(program.get("mode", "candidate_literal"))
    if program_mode not in {"candidate_literal", "task_adapter"}:
        raise ValueError("unsupported program.mode {!r}: {}".format(program_mode, config_path))
    candidate_files = [
        CandidateFile(
            path=str(row["path"]),
            inject_as=str(row["inject_as"]),
            allowed_paths=[str(item) for item in row.get("allowed_paths", [])],
        )
        for row in policy.get("files", [])
    ]
    frozen_inputs = []
    frozen_input_names = set()
    for row in execution.get("frozen_inputs", []):
        if not isinstance(row, dict):
            raise ValueError("execution.frozen_inputs entries must be objects")
        name = str(_required(row, "name", config_path))
        source = str(row.get("source", "baseline_run"))
        relative_path = str(_required(row, "path", config_path))
        inject_as = str(_required(row, "inject_as", config_path))
        required_files = [str(item) for item in row.get("required_files", [])]
        if (
            not name
            or not name.replace("_", "").replace("-", "").isalnum()
            or name in frozen_input_names
        ):
            raise ValueError("execution.frozen_inputs names must be unique and non-empty")
        frozen_input_names.add(name)
        if source not in {"baseline_run", "reference_run"}:
            raise ValueError(
                "unsupported frozen input source {!r}: {}".format(source, config_path)
            )
        for label, value in (("path", relative_path), ("inject_as", inject_as)):
            if not value:
                raise ValueError("frozen input {} cannot be empty".format(label))
        if not required_files:
            raise ValueError("frozen input {} must declare required_files".format(name))
        json_checks = []
        for check in row.get("json_checks", []):
            if not isinstance(check, dict) or not isinstance(check.get("equals"), dict):
                raise ValueError("frozen input json_checks must contain path and equals")
            check_path = str(_required(check, "path", config_path))
            if check_path not in required_files:
                raise ValueError(
                    "frozen input JSON check must target a required file: {}".format(
                        check_path
                    )
                )
            json_checks.append(
                FrozenInputJsonCheck(path=check_path, equals=dict(check["equals"]))
            )
        frozen_inputs.append(
            FrozenInput(
                name=name,
                source=source,
                path=relative_path,
                inject_as=inject_as,
                required_files=required_files,
                json_checks=json_checks,
            )
        )
    spec = TaskSpec(
        root=root,
        task_id=str(_required(data, "id", config_path)),
        title=str(_required(data, "title", config_path)),
        description=str(_required(data, "description", config_path)),
        kind=str(_required(data, "kind", config_path)),
        base_config=(root / str(data.get("base_config", "base_config.yaml"))).resolve(),
        initial_candidate=(root / str(data.get("initial_candidate", "initial_candidate"))).resolve(),
        prompt=(root / str(data.get("prompt", "prompt.md"))).resolve(),
        start_stage=str(_required(execution, "from", config_path)),
        stop_stage=str(_required(execution, "to", config_path)),
        upstream_stages=[str(item) for item in execution.get("upstream_stages", [])],
        evaluation_boundary=evaluation_boundary,
        allowed_config_paths=[str(item) for item in policy.get("allowed_config_paths", [])],
        allowed_source_paths=[str(item) for item in policy.get("allowed_source_paths", [])],
        candidate_files=candidate_files,
        value_constraints=[dict(item) for item in policy.get("value_constraints", [])],
        max_changed_files=int(policy.get("max_changed_files", 0)),
        max_added_lines=int(policy.get("max_added_lines", 0)),
        max_deleted_lines=int(policy.get("max_deleted_lines", 0)),
        prompt_source_ranges=[
            SourceRange(
                path=str(row["path"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
            )
            for row in prompt_context.get("source_ranges", [])
        ],
        prompt_read_only_source_paths=[
            str(item) for item in prompt_context.get("read_only_source_paths", [])
        ],
        priority_parameters=[str(item) for item in prompt_context.get("priority_parameters", [])],
        max_source_chars=int(prompt_context.get("max_source_chars", 80_000)),
        feedback_recent=int(prompt_context.get("feedback_recent", 3)),
        feedback_top=int(prompt_context.get("feedback_top", 3)),
        max_feedback_chars=int(prompt_context.get("max_feedback_chars", 24_000)),
        program_mode=program_mode,
        program_template=(root / str(program["template"])).resolve()
        if program.get("template")
        else None,
        program_adapter=(root / str(program["adapter"])).resolve()
        if program.get("adapter")
        else None,
        program_entrypoint=str(program.get("entrypoint", "evolve_candidate")),
        benchmark_input_id=str(data["benchmark_input"]) if data.get("benchmark_input") else None,
        frozen_inputs=frozen_inputs,
        availability=availability,
        scoring=parse_scoring(
            data.get("scoring"),
            str(_required(data, "kind", config_path)),
            source=str(config_path),
        ),
    )
    prompt_source_paths = set(spec.allowed_source_paths) | set(
        spec.prompt_read_only_source_paths
    )
    if (
        spec.evaluation_boundary == "dreamplace-global-placement"
        and spec.kind
        not in {
            "placement_initialization_source",
            "placement_preconditioner_source",
            "placement_source",
            "placement_parameters",
            "placement_dynamic_parameters",
        }
    ):
        raise ValueError(
            "dreamplace-global-placement boundary is valid only for placement tasks"
        )
    for source_range in spec.prompt_source_ranges:
        if source_range.path not in prompt_source_paths:
            raise ValueError(
                "prompt source range is neither mutable nor declared read-only: {}".format(
                    source_range.path
                )
            )
        if source_range.start_line < 1 or source_range.end_line < source_range.start_line:
            raise ValueError(
                "invalid prompt source range {}:{}-{}".format(
                    source_range.path, source_range.start_line, source_range.end_line
                )
            )
    for name, value in (
        ("max_source_chars", spec.max_source_chars),
        ("feedback_recent", spec.feedback_recent),
        ("feedback_top", spec.feedback_top),
        ("max_feedback_chars", spec.max_feedback_chars),
    ):
        if value < 0:
            raise ValueError("prompt_context.{} cannot be negative".format(name))
    for required_path in (spec.base_config, spec.initial_candidate, spec.prompt):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    if spec.program_mode == "task_adapter":
        if spec.program_template is None or not spec.program_template.is_file():
            raise FileNotFoundError(spec.program_template or root / "<program.template>")
        if spec.program_adapter is None or not spec.program_adapter.is_file():
            raise FileNotFoundError(spec.program_adapter or root / "<program.adapter>")
        if not spec.program_entrypoint.isidentifier():
            raise ValueError("program.entrypoint must be a Python identifier")
    return spec


def require_executable_task(spec: TaskSpec) -> None:
    """A definition is inspectable, but must not silently run a native baseline."""
    if spec.availability == "definition-only":
        raise ValueError(
            "Task {} is definition-only: native policy wiring, trusted result collection "
            "and same-contract reference qualification are not complete".format(spec.task_id)
        )


def load_candidate(candidate_dir: Path) -> Dict[str, Any]:
    path = candidate_dir.expanduser().resolve() / "candidate.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_candidate_payload(data, source=str(path))


def validate_candidate_payload(data: Any, *, source: str = "candidate") -> Dict[str, Any]:
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("schema_version"), int)
        or isinstance(data.get("schema_version"), bool)
        or data.get("schema_version") != 1
    ):
        raise ValueError("{} must be a schema_version 1 object".format(source))
    allowed = {
        "schema_version",
        "config_overrides",
        "candidate_files",
        "source_patch",
        "notes",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("{} contains unknown fields: {}".format(source, ", ".join(unknown)))
    required = {"config_overrides", "candidate_files", "source_patch"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError("{} is missing required fields: {}".format(source, ", ".join(missing)))
    for key in ("config_overrides", "candidate_files"):
        if not isinstance(data.get(key, {}), dict):
            raise ValueError("{}.{} must be an object".format(source, key))
    if not isinstance(data.get("source_patch", ""), str):
        raise ValueError("{}.source_patch must be a string".format(source))
    if not isinstance(data.get("notes", ""), str):
        raise ValueError("{}.notes must be a string".format(source))
    if len(data.get("notes", "")) > 4000:
        raise ValueError("{}.notes exceeds 4000 characters".format(source))
    try:
        encoded = json.dumps(data, allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("{} must contain finite JSON values: {}".format(source, error)) from error
    if len(encoded) > 2_000_000:
        raise ValueError("{} exceeds the 2 MB candidate limit".format(source))
    return dict(data)
