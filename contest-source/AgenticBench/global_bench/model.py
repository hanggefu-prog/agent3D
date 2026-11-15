"""Strict, scope-free task contracts for workspace-oriented global trials."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_POLICIES = frozenset({"quick", "suffix", "final", "full"})


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be an object".format(label))
    return value


def _json_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    """Detach YAML mappings and reject values that cannot enter task identity."""

    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be a finite JSON object".format(label)) from error
    if not isinstance(decoded, dict):
        raise ValueError("{} must be a JSON object".format(label))
    return decoded


def _keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    label: str,
) -> None:
    missing = set(required) - set(value)
    extra = set(value) - set(required) - set(optional)
    if missing or extra:
        raise ValueError(
            "{} has missing={} extra={}".format(label, sorted(missing), sorted(extra))
        )


def _identifier(value: Any, label: str) -> str:
    text = str(value)
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError("{} is not a safe identifier: {!r}".format(label, text))
    return text


def _container_path(value: Any, label: str) -> str:
    path = PurePosixPath(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must be an absolute normalized container path".format(label))
    return path.as_posix()


def _relative_path(value: Any, label: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("{} must be a safe relative path".format(label))
    return path.as_posix()


def _string_tuple(value: Any, label: str, *, allow_empty: bool = True) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("{} must be an array".format(label))
    result = tuple(str(item) for item in value)
    if not allow_empty and not result:
        raise ValueError("{} must not be empty".format(label))
    if any(not item.strip() for item in result):
        raise ValueError("{} contains an empty value".format(label))
    return result


@dataclass(frozen=True)
class SourceSpec:
    repository: str
    revision: str


@dataclass(frozen=True)
class WorkspaceSpec:
    container_path: str
    runs_path: str
    scratch_path: str
    protected_paths: Tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentSpec:
    image: str
    image_digest: Optional[str]
    cpu: float
    memory_gb: float
    storage_gb: float
    gpus: int
    walltime_seconds: int
    network: str
    pids_limit: int


@dataclass(frozen=True)
class InputSpec:
    name: str
    source: str
    mount: str
    read_only: bool
    sha256: Optional[str]


@dataclass(frozen=True)
class SubmissionSpec:
    required_artifacts: Tuple[str, ...]


@dataclass(frozen=True)
class VerifierSpec:
    verifier_id: str
    separate_environment: bool


@dataclass(frozen=True)
class ObjectiveSpec:
    kind: str
    evaluator_profile: Optional[str]
    primary_metric: str
    direction: str
    reference_input: Optional[str]
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class SeedPolicy:
    """Evaluator-owned common-random-number contract."""

    io_placement: int
    partition: int
    placement: int
    routing: int


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    directory: str
    runner_name: str
    depends_on: Tuple[str, ...]
    implementation_paths: Tuple[str, ...]


@dataclass(frozen=True)
class EvaluationSpec:
    stages: Tuple[StageSpec, ...]
    unknown_change_policy: str
    default_policy: str

    def stage_ids(self) -> Tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)


@dataclass(frozen=True)
class BudgetSpec:
    global_rounds: int
    concurrent_trials: int
    expensive_evaluations: int


@dataclass(frozen=True)
class GlobalTaskSpec:
    """A whole-workspace problem; it deliberately contains no source allowlist."""

    schema_version: int
    task_id: str
    title: str
    root: Path
    instruction_path: Path
    instruction: str
    source: SourceSpec
    workspace: WorkspaceSpec
    environment: EnvironmentSpec
    inputs: Tuple[InputSpec, ...]
    submission: SubmissionSpec
    verifier: VerifierSpec
    objective: ObjectiveSpec
    seed_policy: SeedPolicy
    evaluation: EvaluationSpec
    budget: BudgetSpec

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["root"] = str(self.root)
        value["instruction_path"] = str(self.instruction_path)
        return value

    def digest(self) -> str:
        value = self.to_dict()
        value.pop("root", None)
        value.pop("instruction_path", None)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_stage(row: Mapping[str, Any], index: int) -> StageSpec:
    label = "evaluation.stages[{}]".format(index)
    _keys(
        row,
        required=("id", "directory", "runner_name", "depends_on", "implementation_paths"),
        label=label,
    )
    paths = tuple(
        _relative_path(item, label + ".implementation_paths")
        for item in _string_tuple(row["implementation_paths"], label + ".implementation_paths")
    )
    return StageSpec(
        stage_id=_identifier(row["id"], label + ".id"),
        directory=_relative_path(row["directory"], label + ".directory"),
        runner_name=_identifier(row["runner_name"], label + ".runner_name"),
        depends_on=tuple(
            _identifier(item, label + ".depends_on")
            for item in _string_tuple(row["depends_on"], label + ".depends_on")
        ),
        implementation_paths=paths,
    )


def _validate_stage_dag(stages: Tuple[StageSpec, ...]) -> None:
    ids = tuple(stage.stage_id for stage in stages)
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation stage IDs must be unique")
    known = set(ids)
    for stage in stages:
        unknown = set(stage.depends_on) - known
        if unknown:
            raise ValueError("{} depends on unknown stages {}".format(stage.stage_id, sorted(unknown)))
        if stage.stage_id in stage.depends_on:
            raise ValueError("{} cannot depend on itself".format(stage.stage_id))

    visiting = set()
    visited = set()
    by_id = {stage.stage_id: stage for stage in stages}

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            raise ValueError("evaluation stage graph contains a cycle")
        visiting.add(stage_id)
        for dependency in by_id[stage_id].depends_on:
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in ids:
        visit(stage_id)


def load_global_task(task_dir: Path) -> GlobalTaskSpec:
    root = task_dir.expanduser().resolve()
    task_path = root / "task.yaml"
    if not task_path.is_file():
        raise FileNotFoundError("global task is missing task.yaml: {}".format(root))
    data = _mapping(yaml.safe_load(task_path.read_text(encoding="utf-8")), "task.yaml")
    _keys(
        data,
        required=(
            "schema_version",
            "id",
            "title",
            "instruction",
            "source",
            "workspace",
            "environment",
            "inputs",
            "submission",
            "verifier",
            "objective",
            "seed_policy",
            "evaluation",
            "budget",
        ),
        label="task.yaml",
    )
    if int(data["schema_version"]) != 1:
        raise ValueError("unsupported global task schema_version")

    instruction_relative = _relative_path(data["instruction"], "instruction")
    instruction_path = (root / instruction_relative).resolve()
    if root not in instruction_path.parents or not instruction_path.is_file():
        raise ValueError("instruction must resolve to a file inside the task directory")
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError("global task instruction must not be empty")

    source_data = _mapping(data["source"], "source")
    _keys(source_data, required=("repository", "revision"), label="source")
    source = SourceSpec(str(source_data["repository"]), str(source_data["revision"]))
    if not source.repository.strip() or not source.revision.strip():
        raise ValueError("source repository and revision must not be empty")

    workspace_data = _mapping(data["workspace"], "workspace")
    _keys(
        workspace_data,
        required=("container_path", "runs_path", "scratch_path", "protected_paths"),
        label="workspace",
    )
    workspace = WorkspaceSpec(
        _container_path(workspace_data["container_path"], "workspace.container_path"),
        _container_path(workspace_data["runs_path"], "workspace.runs_path"),
        _container_path(workspace_data["scratch_path"], "workspace.scratch_path"),
        tuple(
            _relative_path(item, "workspace.protected_paths")
            for item in _string_tuple(
                workspace_data["protected_paths"], "workspace.protected_paths"
            )
        ),
    )
    if len({workspace.container_path, workspace.runs_path, workspace.scratch_path}) != 3:
        raise ValueError("workspace container paths must be distinct")

    environment_data = _mapping(data["environment"], "environment")
    _keys(
        environment_data,
        required=(
            "image",
            "image_digest",
            "cpu",
            "memory_gb",
            "storage_gb",
            "gpus",
            "walltime_seconds",
            "network",
            "pids_limit",
        ),
        label="environment",
    )
    image_digest = environment_data["image_digest"]
    environment = EnvironmentSpec(
        image=str(environment_data["image"]),
        image_digest=None if image_digest is None else str(image_digest),
        cpu=float(environment_data["cpu"]),
        memory_gb=float(environment_data["memory_gb"]),
        storage_gb=float(environment_data["storage_gb"]),
        gpus=int(environment_data["gpus"]),
        walltime_seconds=int(environment_data["walltime_seconds"]),
        network=str(environment_data["network"]),
        pids_limit=int(environment_data["pids_limit"]),
    )
    if (
        not environment.image.strip()
        or environment.cpu <= 0
        or environment.memory_gb <= 0
        or environment.storage_gb <= 0
        or environment.gpus < 0
        or environment.walltime_seconds <= 0
        or environment.pids_limit <= 0
    ):
        raise ValueError("environment resources must be positive")
    if environment.network not in {"none", "restricted", "host"}:
        raise ValueError("environment.network must be none, restricted, or host")

    if not isinstance(data["inputs"], list):
        raise ValueError("inputs must be an array")
    inputs = []
    for index, raw in enumerate(data["inputs"]):
        row = _mapping(raw, "inputs[{}]".format(index))
        _keys(row, required=("name", "source", "mount", "read_only", "sha256"), label="inputs")
        sha256 = row["sha256"]
        item = InputSpec(
            name=_identifier(row["name"], "inputs.name"),
            source=str(row["source"]),
            mount=_container_path(row["mount"], "inputs.mount"),
            read_only=bool(row["read_only"]),
            sha256=None if sha256 is None else str(sha256),
        )
        if not item.read_only:
            raise ValueError("global task inputs must be read-only")
        inputs.append(item)
    if len({item.name for item in inputs}) != len(inputs):
        raise ValueError("input names must be unique")

    submission_data = _mapping(data["submission"], "submission")
    _keys(submission_data, required=("required_artifacts",), label="submission")
    submission = SubmissionSpec(
        tuple(
            _identifier(item, "submission.required_artifacts")
            for item in _string_tuple(
                submission_data["required_artifacts"],
                "submission.required_artifacts",
                allow_empty=False,
            )
        )
    )

    verifier_data = _mapping(data["verifier"], "verifier")
    _keys(verifier_data, required=("id", "separate_environment"), label="verifier")
    verifier = VerifierSpec(
        _identifier(verifier_data["id"], "verifier.id"),
        bool(verifier_data["separate_environment"]),
    )
    if not verifier.separate_environment:
        raise ValueError("global task verifier must run in a separate environment")

    objective_data = _mapping(data["objective"], "objective")
    _keys(
        objective_data,
        required=(
            "kind",
            "evaluator_profile",
            "primary_metric",
            "direction",
            "reference_input",
            "parameters",
        ),
        label="objective",
    )
    evaluator_profile = objective_data["evaluator_profile"]
    reference_input = objective_data["reference_input"]
    objective = ObjectiveSpec(
        kind=_identifier(objective_data["kind"], "objective.kind"),
        evaluator_profile=(
            None
            if evaluator_profile is None
            else _identifier(evaluator_profile, "objective.evaluator_profile")
        ),
        primary_metric=_identifier(
            objective_data["primary_metric"], "objective.primary_metric"
        ),
        direction=str(objective_data["direction"]),
        reference_input=(
            None
            if reference_input is None
            else _identifier(reference_input, "objective.reference_input")
        ),
        parameters=_json_mapping(
            _mapping(objective_data["parameters"], "objective.parameters"),
            "objective.parameters",
        ),
    )
    if objective.direction not in {"minimize", "maximize"}:
        raise ValueError("objective.direction must be minimize or maximize")
    input_names = {item.name for item in inputs}
    if objective.reference_input is not None and objective.reference_input not in input_names:
        raise ValueError("objective.reference_input is not declared under inputs")
    if objective.kind == "sta_wns_tns":
        if objective.direction != "maximize" or objective.evaluator_profile is None:
            raise ValueError("STA objective must maximize a declared evaluator profile")
        required_parameters = {"wns_weight", "tns_weight", "deficit_floor", "opt_passes"}
        if set(objective.parameters) != required_parameters:
            raise ValueError("STA objective parameters must be {}".format(sorted(required_parameters)))
        wns_weight = float(objective.parameters["wns_weight"])
        tns_weight = float(objective.parameters["tns_weight"])
        if wns_weight <= 0 or tns_weight <= 0 or abs(wns_weight + tns_weight - 1.0) > 1e-12:
            raise ValueError("STA objective weights must be positive and sum to one")
        if float(objective.parameters["deficit_floor"]) <= 0:
            raise ValueError("STA deficit floor must be positive")
        profile_passes = {"tp_gnn_placement_two_pass": 2, "tp_gnn_placement_one_pass": 1,
                          "tp_gnn_placement_one_pass_20261004": 1}
        required_passes = profile_passes.get(objective.evaluator_profile)
        if (required_passes is None or type(objective.parameters["opt_passes"]) is not int
                or objective.parameters["opt_passes"] != required_passes):
            raise ValueError("STA co-design OPT pass count must match its versioned evaluator profile")
    elif objective.kind == "hotspot_average_temperature":
        if objective.evaluator_profile is None:
            raise ValueError("thermal objective must declare an evaluator profile")
        required_parameters = {"metric_path", "ambient_k", "grid_size"}
        if objective.evaluator_profile == "tp_gnn_placement_cts_thermal_20261008":
            if objective.direction != "maximize" or objective.primary_metric != "combined_score":
                raise ValueError("Latest thermal objective must maximize the timing-penalized combined_score")
            timing_parameters = {"timing_endpoint": "post_cts_one_opt", "wns_weight": 0.6,
                                 "tns_weight": 0.4, "deficit_floor": 1.0e-12,
                                 "timing_tolerance_fraction": 0.05,
                                 "timing_penalty": "weighted_excess_cost_additive_v1",
                                 "timing_penalty_weight": 0.1}
            required_parameters.update(timing_parameters)
            for name, expected in timing_parameters.items():
                value = objective.parameters.get(name)
                if isinstance(value, bool) or value != expected:
                    raise ValueError("Latest thermal timing parameter {} must be {}".format(name, expected))
            required_parameters.add("opt_passes")
            if type(objective.parameters.get("opt_passes")) is not int or objective.parameters["opt_passes"] != 1:
                raise ValueError("Latest CTS thermal endpoint requires exactly one OPT pass")
        elif objective.direction != "minimize":
            raise ValueError("Legacy thermal objective must minimize temperature")
        if set(objective.parameters) != required_parameters:
            raise ValueError(
                "thermal objective parameters must be {}".format(sorted(required_parameters))
            )
        if objective.parameters["metric_path"] != "temperature.global_average_k":
            raise ValueError("thermal objective must use HotSpot global average temperature")
        if float(objective.parameters["ambient_k"]) <= 0 or int(objective.parameters["grid_size"]) < 2:
            raise ValueError("thermal ambient and grid size are invalid")
    elif objective.kind == "user_defined_ppa":
        if objective.evaluator_profile is not None or objective.reference_input is not None:
            raise ValueError("user-defined PPA objective cannot bind a fixed profile/reference")
    else:
        raise ValueError("unsupported global objective kind: {}".format(objective.kind))

    seed_data = _mapping(data["seed_policy"], "seed_policy")
    _keys(
        seed_data,
        required=("io_placement", "partition", "placement", "routing"),
        label="seed_policy",
    )
    seed_policy = SeedPolicy(
        io_placement=int(seed_data["io_placement"]),
        partition=int(seed_data["partition"]),
        placement=int(seed_data["placement"]),
        routing=int(seed_data["routing"]),
    )
    if min(seed_policy.io_placement, seed_policy.partition, seed_policy.placement) < 0:
        raise ValueError("I/O, partition, and placement seeds must be non-negative")
    if seed_policy.routing < -1:
        raise ValueError("routing seed must be -1 or non-negative")

    evaluation_data = _mapping(data["evaluation"], "evaluation")
    _keys(
        evaluation_data,
        required=("unknown_change_policy", "default_policy", "stages"),
        label="evaluation",
    )
    if evaluation_data["unknown_change_policy"] not in {"all", "reject"}:
        raise ValueError("unknown_change_policy must be all or reject")
    if evaluation_data["default_policy"] not in _POLICIES:
        raise ValueError("unsupported evaluation.default_policy")
    if not isinstance(evaluation_data["stages"], list) or not evaluation_data["stages"]:
        raise ValueError("evaluation.stages must be a non-empty array")
    stages = tuple(
        _parse_stage(_mapping(row, "evaluation stage"), index)
        for index, row in enumerate(evaluation_data["stages"])
    )
    _validate_stage_dag(stages)
    evaluation = EvaluationSpec(
        stages=stages,
        unknown_change_policy=str(evaluation_data["unknown_change_policy"]),
        default_policy=str(evaluation_data["default_policy"]),
    )

    budget_data = _mapping(data["budget"], "budget")
    _keys(
        budget_data,
        required=("global_rounds", "concurrent_trials", "expensive_evaluations"),
        label="budget",
    )
    budget = BudgetSpec(
        int(budget_data["global_rounds"]),
        int(budget_data["concurrent_trials"]),
        int(budget_data["expensive_evaluations"]),
    )
    if min(budget.global_rounds, budget.concurrent_trials, budget.expensive_evaluations) < 1:
        raise ValueError("budget values must be positive")

    return GlobalTaskSpec(
        schema_version=1,
        task_id=_identifier(data["id"], "id"),
        title=str(data["title"]),
        root=root,
        instruction_path=instruction_path,
        instruction=instruction,
        source=source,
        workspace=workspace,
        environment=environment,
        inputs=tuple(inputs),
        submission=submission,
        verifier=verifier,
        objective=objective,
        seed_policy=seed_policy,
        evaluation=evaluation,
        budget=budget,
    )
