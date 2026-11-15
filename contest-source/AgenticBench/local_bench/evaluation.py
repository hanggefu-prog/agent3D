from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from .metrics import extract_metrics, _source_grt_metrics_path
from .tp_gnn_contract import TP_GNN_METHODS
from .model import (
    ROUTING_SOURCE_TASK_KINDS,
    EvaluationResult,
    FrozenInput,
    TaskSpec,
    load_candidate,
    load_task,
)
from .policy import validate_patch_scope
from .verifier import verify_candidate
from .workspace import (
    apply_candidate_patch,
    copy_upstream_stages,
    isolated_clone,
    materialize_dreamplace_source_tree,
    prepare_dreamplace_overlay,
    prepare_openroad_runtime,
    validate_candidate_workspace,
)


PLACEMENT_TASK_KINDS = {
    "placement_initialization_source",
    "placement_preconditioner_source",
    "placement_source",
    "placement_parameters",
    "placement_dynamic_parameters",
    "placement_two_pass_sta",
}

REFERENCE_PLACEMENT_TASKS = {
    "placement_dynamic_parameters_reference_20260923",
    "placement_preconditioner_source_reference_20260923",
}

COMPONENT_PLACEMENT_TASKS = {"placement_initialization", "placement_preconditioner"}

STA_TWO_PASS_TASK_KINDS = {
    "tp_gnn_placement_sta",
    "placement_two_pass_sta",
    "tp_gnn_placement_cts_thermal",
}

THERMAL_TASK_KINDS = {"tp_gnn_placement_thermal", "tp_gnn_placement_cts_thermal"}

# Randomness belongs to the benchmark contract, never to a candidate.  These
# values are injected after candidate verification so every reference and
# candidate is evaluated with common random numbers.
FIXED_BENCHMARK_SEEDS = {
    "pins": 1,
    "partition": 1001,
    "placement": 1000,
    "routing": -1,
}

UPSTREAM_STAGE_DIRS = {
    "input": "00_input",
    "floorplan": "10_floorplan",
    "partition": "20_partition",
    "placement": "30_placement",
    "pdn": "40_pdn",
    "hbt": "50_hbt",
    "opt": "60_opt",
    "cts": "70_cts",
}

EVALUATOR_SOURCE_PATCHES = {
    "tp_gnn_placement_cts_thermal": (
        Path(__file__).resolve().parent / "evaluator_patches" / "tp_gnn_placement_sta.patch"
    ),
    "tp_gnn_placement_sta": (
        Path(__file__).resolve().parent
        / "evaluator_patches"
        / "tp_gnn_placement_sta.patch"
    ),
    # Placement bests were produced against the frozen 2026-08-30 source
    # tree.  This task-specific trusted patch advances only the evaluator-owned
    # OPT implementation to the validated two-pass STA contract; the selected
    # placement patch still applies to its exact original source baseline.
    "placement_two_pass_sta": (
        Path(__file__).resolve().parent
        / "evaluator_patches"
        / "placement_two_pass_sta_clean_clone.patch"
    ),
}


def select_evaluator_source_patch(kind: str, workspace: Path) -> Optional[Path]:
    """Select a trusted adapter by applicability, never by candidate preference.

    Historical placement promotion and current co-design share the same two-pass
    objective but have different core interfaces. Check the exact source bytes
    before applying either adapter. An unknown or ambiguous interface fails
    before launching physical tools; it must not silently run unpatched OPT.
    """
    patch = EVALUATOR_SOURCE_PATCHES.get(kind)
    if patch is None:
        return None
    candidates = [patch]
    if kind == "placement_two_pass_sta":
        candidates = [EVALUATOR_SOURCE_PATCHES["tp_gnn_placement_sta"], patch]
    applicable, errors = [], []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            raise FileNotFoundError("trusted evaluator source patch is missing: {}".format(candidate))
        result = subprocess.run(
            ["git", "apply", "--check", str(candidate)], cwd=str(workspace),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode == 0:
            applicable.append(candidate)
        else:
            errors.append("{}: {}".format(candidate.name, result.stderr.strip()))
    if len(applicable) != 1:
        raise ValueError(
            "trusted two-pass adapter requires exactly one compatible source interface "
            "for {}; found {}. {}".format(kind, len(applicable), "; ".join(errors))
        )
    return applicable[0]


def _replay_start_stage(spec: TaskSpec, config: Mapping[str, Any], plan_output: str) -> str:
    """Adopt frozen timing through its real producer, without recomputing STA.

    Older profiles start at partition and provide timing_input separately. The
    maintained flow requires a signed step-15 handoff before step 20. Starting
    at step 15 asks that stage to validate/copy the already frozen feature set;
    it does not invent a manifest or relabel an old stage as a new one.
    """
    if spec.start_stage != "partition" or not config.get("partition", {}).get("timing_input"):
        return spec.start_stage
    try:
        stages = json.loads(plan_output)["plan"]
    except (ValueError, KeyError, TypeError):
        raise ValueError("cannot inspect timing-input handoff in the trusted flow plan")
    if any(row.get("stage") == "timing_extract" and row.get("skip") is None for row in stages):
        return "timing_extract"
    return spec.start_stage  # Historical flows without a distinct step 15.


def _merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_dotted(data: Dict[str, Any], dotted: str, value: Any) -> None:
    cursor = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError("cannot inject {} through non-object {}".format(dotted, part))
        cursor = child
    cursor[parts[-1]] = value


def _write_resolved_config(
    spec: TaskSpec,
    candidate_dir: Path,
    output: Path,
    candidate: Dict[str, Any],
    base_config: Optional[Path] = None,
    seed_schedule: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    selected_base = (base_config or spec.base_config).expanduser().resolve()
    if not selected_base.is_file():
        raise FileNotFoundError("base_config: {}".format(selected_base))
    base = yaml.safe_load(selected_base.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base_config must contain a mapping: {}".format(selected_base))
    seeds = dict(FIXED_BENCHMARK_SEEDS)
    if seed_schedule is not None:
        from .seeds import validate_schedule
        schedule = validate_schedule(seed_schedule)
        seeds = {"pins": schedule["io"], **{k: schedule[k] for k in ("partition", "placement", "routing")}}
    resolved = _merge(base, candidate.get("config_overrides", {}))
    _set_dotted(resolved, "pins.seed", seeds["pins"])
    _set_dotted(resolved, "partition.seed", seeds["partition"])
    _set_dotted(resolved, "placement.seed", seeds["placement"])
    _set_dotted(resolved, "routing.seed", seeds["routing"])
    if spec.kind in THERMAL_TASK_KINDS:
        # The focused thermal adapter shares the STA frontend baseline. Never
        # inherit the general LoL Mt-KaHyPar default when a selector is omitted.
        # Keep the exact explicitly selected algorithm. The root accepts the
        # old name as an input alias, while result/reference validation checks
        # native identity and never treats old and flat artifacts as equal.
        partition_values = resolved.get("partition", {})
        if not isinstance(partition_values, Mapping):
            raise ValueError("thermal co-design requires partition configuration")
        partition_method = partition_values.get("algorithm", "tp-gnn-flat")
        if partition_method not in TP_GNN_METHODS:
            raise ValueError("thermal co-design requires TP-GNN partitioning")
        for stage, method in (("partition", partition_method), ("placement", "concurrent-dreamplace")):
            config = resolved.setdefault(stage, {})
            if not isinstance(config, dict) or config.get("algorithm", method) != method:
                raise ValueError("thermal co-design requires {}.algorithm={}".format(stage, method))
            config["algorithm"] = method
        # This is evaluator-owned configuration, not a candidate knob.
        resolved["thermal"] = {
            "enabled": True,
            "grid_size": 10,
            "ambient_k": 318.15,
        }
    if spec.kind in ROUTING_SOURCE_TASK_KINDS:
        routing = resolved.get("routing", {})
        iterations = (
            routing.get("congestion_iterations")
            if isinstance(routing, Mapping)
            else None
        )
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 4:
            raise ValueError(
                "routing source evolution requires congestion_iterations >= 4 "
                "so every active policy receives a meaningful fixed reroute budget"
            )
    embedded = candidate.get("candidate_files", {})
    if not isinstance(embedded, dict):
        raise ValueError("candidate.candidate_files must be an object")
    declared = {item.path for item in spec.candidate_files}
    unknown = sorted(set(embedded) - declared)
    if unknown:
        raise ValueError("candidate contains undeclared files: {}".format(", ".join(unknown)))
    for item in spec.candidate_files:
        if item.path in embedded:
            contents = deepcopy(embedded[item.path])
            if not isinstance(contents, dict):
                raise ValueError("candidate file must contain a JSON object: {}".format(item.path))
            path = (output.parent / "candidate_files" / item.path).resolve()
        else:
            source = (candidate_dir / item.path).resolve()
            if not source.is_file() or candidate_dir.resolve() not in source.parents:
                raise ValueError("candidate file is missing or escapes candidate directory: {}".format(item.path))
            contents = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(contents, dict):
                raise ValueError("candidate file must contain a JSON object: {}".format(item.path))
            path = (output.parent / "candidate_files" / item.path).resolve()
        if item.path == "partition.json":
            _set_dotted(contents, "partition.seed", seeds["partition"])
        elif item.path == "placement.json":
            _set_dotted(
                contents,
                "concurrent_dreamplace.initialization_seed",
                seeds["placement"],
            )
            if spec.task_id in COMPONENT_PLACEMENT_TASKS:
                # Trusted physical inputs, not model-owned search variables.
                halo = resolved["placement"]["macro_halo_um"]
                for axis in ("x", "y"):
                    _set_dotted(contents, "concurrent_dreamplace.macro_halo_" + axis, halo)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(contents, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _set_dotted(resolved, item.inject_as, str(path))
    # Parameterized placement tasks inject a candidate-owned placement.json
    # and must keep its internal density synchronized with the public config.
    # Source-only placement tasks deliberately declare no such file; they use
    # the frozen base configuration while evolving only implementation code.
    if spec.kind in PLACEMENT_TASK_KINDS and "placement.json" in declared:
        placement_file = embedded.get("placement.json", {})
        algorithm_values = (
            placement_file.get("concurrent_dreamplace", {})
            if isinstance(placement_file, dict)
            else {}
        )
        algorithm_density = algorithm_values.get("target_density")
        public_density = resolved.get("placement", {}).get("target_density")
        if algorithm_density is None or public_density is None:
            raise ValueError("placement parameter candidate must define both target-density fields")
        if abs(float(algorithm_density) - float(public_density)) > 1.0e-12:
            raise ValueError(
                "placement.target_density and concurrent_dreamplace.target_density must match"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return resolved


def _validate_placement_partition_reference(
    spec: TaskSpec, baseline_run: Path, base_config: Optional[Path]
) -> Dict[str, Any]:
    """Bind frozen placement input to the trusted, pre-override producer.

    A selected input bundle can replace the task's historical default profile.
    Its base configuration is supplied by the trusted caller, not materialized
    from candidate overrides. Keep old and flat native identities distinct.
    """
    selected_base = (base_config or spec.base_config).expanduser().resolve()
    base = yaml.safe_load(selected_base.read_text(encoding="utf-8"))
    partition = base.get("partition") if isinstance(base, dict) else None
    expected = partition.get("algorithm") if isinstance(partition, dict) else None
    if expected not in TP_GNN_METHODS:
        raise ValueError("placement reference requires an explicit trusted TP-GNN producer identity")
    from .placement_baseline import validate_tp_gnn_partition

    method = validate_tp_gnn_partition(
        baseline_run, expected_method=expected,
        require_bound_report=expected == "tp-gnn-flat",
        expected_decoder="global-2way",
    )
    return {"status": "passed", "partition_method": method,
            "base_config": str(selected_base),
            "base_config_sha256": _file_sha256(selected_base),
            "hash_bound_report_required": expected == "tp-gnn-flat"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_routing_source_reference(
    spec: TaskSpec,
    baseline_run: Optional[Path],
    reference_run: Optional[Path],
    resolved_config: Mapping[str, Any],
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Bind a routing score to the same frozen input and routing budget."""

    if reference_run is None:
        raise ValueError(
            "full routing source evaluation requires a frozen reference run"
        )
    from .reference_latest import baseline_for_config, BASELINE_ID
    selected = baseline_for_config(resolved_config)
    if selected.BASELINE_ID == BASELINE_ID:
        return selected.validate_grt_reference(reference_run, resolved_config, baseline_run, spec.upstream_stages,
            hook_spec=spec, source_root=source_root)
    reference = reference_run.expanduser().resolve()
    metrics_path = _source_grt_metrics_path(reference)
    if not metrics_path.is_file():
        raise ValueError("routing source reference has no metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metrics, dict)
        or metrics.get("status") != "success"
        or metrics.get("stage") not in {"route-grt", "global_route"}
        or (metrics.get("stage") == "route-grt" and metrics.get("algorithm")
            not in {"openroad-die-local-net-order", "openroad-fastroute-die-local"})
    ):
        raise ValueError("routing source reference is not a successful GRT-only run")
    dies = metrics.get("dies")
    if not isinstance(dies, dict) or set(dies) != {"bottom", "top"}:
        raise ValueError("routing source reference must contain bottom and top dies")
    configuration = metrics.get("configuration")
    if not isinstance(configuration, dict) or (
        metrics.get("stage") == "route-grt"
        and configuration.get("grt_only") is not True
    ):
        raise ValueError("routing source reference did not use the GRT-only boundary")
    routing = resolved_config.get("routing")
    if not isinstance(routing, Mapping):
        raise ValueError("resolved routing configuration is missing")
    expected = {
        "congestion_iterations": int(routing.get("congestion_iterations", -1)),
        "resource_reduction": float(routing.get("resource_reduction", 0.0)),
        "threads": min(int(routing.get("threads", resolved_config.get("threads", 1))), 32),
    }
    observed = {
        "congestion_iterations": int(configuration.get("congestion_iterations", -1)),
        "resource_reduction": float(configuration.get("resource_reduction", -1.0)),
        "threads": int(configuration.get("threads", -1)),
    }
    if observed != expected:
        raise ValueError(
            "routing source reference budget mismatch: expected {}, found {}".format(
                expected, observed
            )
        )
    manifest_hashes: Dict[str, str] = {}
    if spec.upstream_stages:
        if baseline_run is None:
            raise ValueError("routing source reference validation requires frozen input")
        baseline = baseline_run.expanduser().resolve()
        for stage in spec.upstream_stages:
            directory = UPSTREAM_STAGE_DIRS.get(stage)
            if directory is None:
                raise ValueError("unknown frozen upstream stage: {}".format(stage))
            frozen_manifest = baseline / directory / "stage_manifest.json"
            reference_manifest = reference / directory / "stage_manifest.json"
            if not frozen_manifest.is_file() or not reference_manifest.is_file():
                raise ValueError(
                    "routing source reference is missing {} stage provenance".format(stage)
                )
            frozen_hash = _file_sha256(frozen_manifest)
            reference_hash = _file_sha256(reference_manifest)
            if frozen_hash != reference_hash:
                raise ValueError(
                    "routing source reference does not match frozen {} stage".format(stage)
                )
            manifest_hashes[stage] = frozen_hash
    return {
        "status": "passed",
        "metrics": str(metrics_path),
        "routing_budget": expected,
        "upstream_manifest_sha256": manifest_hashes,
    }


# Backward-compatible private name used by existing benchmark tests and older
# experiment scripts.
_validate_routing_net_order_reference = _validate_routing_source_reference


def _validate_sta_codesign_reference(
    spec: TaskSpec,
    baseline_run: Optional[Path],
    reference_run: Optional[Path],
) -> Dict[str, Any]:
    """Require a successful two-pass reference from the same frozen front end."""

    if reference_run is None:
        return {
            "status": "bootstrap",
            "score_semantics": "valid two-pass run becomes the frozen STA reference",
        }
    reference = reference_run.expanduser().resolve()
    metrics_path = reference / "60_opt" / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError("STA co-design reference has no 60_opt/metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or metrics.get("status") != "success":
        raise ValueError("STA co-design reference is not a successful OPT run")
    if int(metrics.get("passes_run", -1)) != 2 or int(
        metrics.get("iterations_run", -1)
    ) != 2:
        raise ValueError("STA co-design reference did not run exactly two OPT passes")
    validity = metrics.get("timing_validity")
    if (
        not isinstance(validity, Mapping)
        or validity.get("graph_completeness") != "passed"
        or validity.get("authoritative_timing_view") != "continuous-passive-hbt"
        or int(validity.get("unclocked_or_unconstrained_endpoints", -1)) != 0
    ):
        raise ValueError("STA co-design reference failed timing-validity gates")

    if baseline_run is None:
        raise ValueError("STA co-design reference validation requires frozen input")
    baseline = baseline_run.expanduser().resolve()
    manifest_hashes: Dict[str, str] = {}
    for stage in spec.upstream_stages:
        directory = UPSTREAM_STAGE_DIRS.get(stage)
        if directory is None:
            raise ValueError("unknown frozen upstream stage: {}".format(stage))
        frozen_manifest = baseline / directory / "stage_manifest.json"
        reference_manifest = reference / directory / "stage_manifest.json"
        if not frozen_manifest.is_file() or not reference_manifest.is_file():
            raise ValueError(
                "STA co-design reference is missing {} provenance".format(stage)
            )
        frozen_hash = _file_sha256(frozen_manifest)
        if frozen_hash != _file_sha256(reference_manifest):
            raise ValueError(
                "STA co-design reference does not match frozen {} stage".format(stage)
            )
        manifest_hashes[stage] = frozen_hash
    return {
        "status": "passed",
        "metrics": str(metrics_path),
        "opt_passes": 2,
        "upstream_manifest_sha256": manifest_hashes,
    }


def _relative_path(root: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("{} must be a safe relative path: {}".format(label, value))
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("{} escapes its source root: {}".format(label, value)) from error
    return resolved


def _lookup_json_field(data: Mapping[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError("JSON assertion field is missing: {}".format(dotted))
        value = value[part]
    return value


def _directory_fingerprint(root: Path) -> Dict[str, Dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("frozen input cannot contain symlinks: {}".format(path))
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    if not records:
        raise ValueError("frozen input directory is empty: {}".format(root))
    return records


def _validate_frozen_input(source: Path, contract: FrozenInput) -> Dict[str, Dict[str, Any]]:
    for required in contract.required_files:
        path = _relative_path(source, required, label="frozen input required file")
        if not path.is_file():
            raise FileNotFoundError(path)
    for check in contract.json_checks:
        path = _relative_path(source, check.path, label="frozen input JSON check")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("frozen input JSON must contain an object: {}".format(path))
        for field, expected in check.equals.items():
            actual = _lookup_json_field(payload, str(field))
            if actual != expected:
                raise ValueError(
                    "frozen input assertion failed for {}:{}: expected {!r}, got {!r}".format(
                        path, field, expected, actual
                    )
                )
    return _directory_fingerprint(source)


def _materialize_frozen_inputs(
    spec: TaskSpec,
    resolved_config: Dict[str, Any],
    resolved_config_path: Path,
    output_dir: Path,
    *,
    baseline_run: Optional[Path],
    reference_run: Optional[Path],
) -> Dict[str, Any]:
    """Copy declared immutable inputs into a candidate-owned, hashed bundle.

    Task definitions describe the source, injection point and semantic checks;
    the common harness owns path safety, adoption, hashing and provenance.
    """

    adopted: Dict[str, Any] = {}
    roots = {"baseline_run": baseline_run, "reference_run": reference_run}
    for contract in spec.frozen_inputs:
        selected = roots[contract.source]
        if selected is None:
            raise ValueError(
                "task {} requires {} for frozen input {}".format(
                    spec.task_id, contract.source, contract.name
                )
            )
        run_root = selected.expanduser().resolve()
        source = _relative_path(run_root, contract.path, label="frozen input source path")
        source_records = _validate_frozen_input(source, contract)
        destination = output_dir / "harness_inputs" / "frozen" / contract.name
        if destination.exists():
            raise FileExistsError("frozen input destination already exists: {}".format(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        copied_records = _directory_fingerprint(destination)
        if copied_records != source_records:
            raise RuntimeError("frozen input changed while being adopted: {}".format(source))
        _set_dotted(resolved_config, contract.inject_as, str(destination))
        adopted[contract.name] = {
            "source": contract.source,
            "source_path": str(source),
            "adopted_path": str(destination),
            "inject_as": contract.inject_as,
            "files": copied_records,
        }
    if adopted:
        resolved_config_path.write_text(
            yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
        )
    return adopted


def _failure(error: Exception, artifacts: Dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        metrics={"combined_score": 0.0, "runs_successfully": 0.0},
        artifacts={
            **artifacts,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def _run_native_flow(
    command: List[str], *, invocation: str, artifacts: Dict[str, Any], **kwargs: Any
) -> subprocess.CompletedProcess:
    """Record a native-flow invocation without claiming individual tool starts.

    A completed child (including a nonzero exit) or ``TimeoutExpired`` proves
    that the flow process started. A launch ``OSError`` proves it did not;
    other exceptions leave launch state unknown. This receipt does not prove
    that an OpenROAD/DREAMPlace/HotSpot process inside that flow was reached.
    Planning, compilation and validation are deliberately not counted here.
    """
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "invocation": invocation,
        "command": list(command),
        "cwd": str(kwargs["cwd"]),
        "timeout_seconds": kwargs.get("timeout"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "wall_seconds": None,
        "process_started": None,
        "returncode": None,
        "status": "dispatching",
    }
    artifacts["native_flow_execution"].append(receipt)
    started = time.monotonic()
    try:
        proc = subprocess.run(command, **kwargs)
    except subprocess.TimeoutExpired as error:
        receipt.update(process_started=True, status="timeout", error_type=type(error).__name__)
        raise
    except OSError as error:
        receipt.update(process_started=False, status="launch_failed", error_type=type(error).__name__)
        raise
    except BaseException as error:
        receipt.update(status="uncertain_exception", error_type=type(error).__name__)
        raise
    else:
        receipt.update(
            process_started=True,
            returncode=proc.returncode,
            status="completed" if proc.returncode == 0 else "failed",
        )
        return proc
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()


def _openroad_runtime_environment(environment: Mapping[str, str]) -> Dict[str, str]:
    """Prefer the C++ runtime paired with the compiler that built OpenROAD.

    Some evaluation images put an older Conda libstdc++ ahead of the system
    compiler runtime.  The build then succeeds with the system compiler but the
    resulting binary cannot start.  A site may provide an exact override; the
    default otherwise derives libstdc++ from the active C++ compiler and keeps
    the bundled OR-Tools directory when present.
    """

    result = dict(environment)
    override = os.environ.get("OPEN3D_LOCAL_OPENROAD_RUNTIME_LIBRARY_PATH")
    if override:
        result["LD_LIBRARY_PATH"] = override
        return result
    directories: List[str] = []
    try:
        compiler = subprocess.run(
            ["c++", "-print-file-name=libstdc++.so.6"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
        library = Path(compiler.stdout.strip()).expanduser()
        if compiler.returncode == 0 and library.is_absolute() and library.is_file():
            directories.append(str(library.resolve().parent))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    for directory in (Path("/usr/lib/x86_64-linux-gnu"), Path("/lib/x86_64-linux-gnu"), Path("/opt/or-tools/lib")):
        if directory.is_dir() and str(directory) not in directories:
            directories.append(str(directory))
    if directories:
        result["LD_LIBRARY_PATH"] = ":".join(directories)
    return result


LATEST_REFERENCE_PDK_SHA256 = "44ce3868364656867e012aa6b6624c1c4fc0f0d260171a38e1472dc06a0feeec"


def _latest_reference_snap_guard(workspace: Path, resolved_config: Mapping[str, Any]) -> float:
    """Read, but never execute, the pinned latest-family PDK implementation.

    The reference family fixes ng45 and its physical source. This small literal
    contract is byte-bound to that source, with parity tested against get_pdk.
    A candidate cannot select a module through PYTHONPATH or substitute a PDK.
    """
    from .reference_latest import PARTITION_CONFIG
    trusted_partition = Path(__file__).resolve().parent / "reference_configs/partition.json"
    if (resolved_config.get("pdk") != "ng45"
            or resolved_config.get("placement", {}).get("macro_halo_um") != 10.0
            or resolved_config.get("partition", {}).get("config") not in
                (PARTITION_CONFIG, str(trusted_partition))):
        raise ValueError("Latest placement PDK/reference-family contract differs")
    workspace = workspace.resolve(strict=True)
    path = workspace / "open3dbench/pdks.py"
    if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
            or path.resolve().parent != workspace / "open3dbench"
            or _file_sha256(path) != LATEST_REFERENCE_PDK_SHA256):
        raise ValueError("Latest placement requires the exact frozen PDK module bytes")
    return max(0.14, 1.0, 0.0) + 0.001


def _dreamplace_global_placement_command(
    spec: TaskSpec,
    resolved_config: Mapping[str, Any],
    workspace: Path,
    run_dir: Path,
    python: str,
    base_params_override: Optional[Path] = None,
    reference_family: Optional[str] = None,
) -> Tuple[List[str], Path]:
    """Build the fast placement-only command without entering die initialization.

    This calls Place-3D's maintained public placement module directly.  The
    Open3DBench flow remains the owner of input, floorplan, and partition
    artifacts, while the benchmark intentionally stops before OpenROAD ODB
    construction and the rest of the physical backend.
    """

    if spec.kind not in PLACEMENT_TASK_KINDS:
        raise ValueError(
            "dreamplace-global-placement boundary requires a placement task"
        )
    if resolved_config.get("type") != "lol":
        raise ValueError(
            "dreamplace-global-placement currently supports fixed-partition LoL"
        )
    placement = resolved_config.get("placement", {})
    input_config = resolved_config.get("input", {})
    if not isinstance(placement, Mapping) or not isinstance(input_config, Mapping):
        raise ValueError("resolved placement/input configuration must be mappings")
    if placement.get("algorithm") != "concurrent-dreamplace":
        raise ValueError(
            "dreamplace-global-placement requires placement.algorithm "
            "concurrent-dreamplace"
        )
    flow_config_value = placement.get(
        "config", "config/placement/concurrent_dreamplace/default.json"
    )
    flow_config = Path(str(flow_config_value)).expanduser()
    if not flow_config.is_absolute():
        flow_config = workspace / "Place-3D" / flow_config
    flow_config = flow_config.resolve()
    if not flow_config.is_file():
        raise FileNotFoundError("placement.config: {}".format(flow_config))

    partition_dir = run_dir / "20_partition"
    base_params = (
        base_params_override.resolve()
        if base_params_override is not None
        else run_dir / "10_floorplan" / "design_params.json"
    )
    for required in (
        partition_dir / "partition_assignment.csv",
        partition_dir / "die_regions.json",
        base_params,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                "global-placement input is missing from the frozen upstream run: {}".format(
                    required
                )
            )

    command = [
        python,
        "-m",
        "src.placement.concurrent_dreamplace",
        "--design",
        str(resolved_config["design"]),
        "--variant",
        str(input_config.get("variant", "default")),
        "--partition-dir",
        str(partition_dir),
        "--output-dir",
        str(run_dir / "30_placement"),
        "--flow-config",
        str(flow_config),
        "--base-params",
        str(base_params),
        "--gpu",
        "1" if placement.get("gpu", True) else "0",
        "--target-density",
        str(placement["target_density"]),
    ]
    if spec.task_id in REFERENCE_PLACEMENT_TASKS | COMPONENT_PLACEMENT_TASKS:
        # Match the native runner's PDK-derived macro snap guard and halo
        # provenance. The legacy placement-only task remains unchanged.
        if spec.task_id in COMPONENT_PLACEMENT_TASKS:
            from open3dbench.pdks import get_pdk
            pdk = get_pdk(resolved_config["pdk"])
            guard = max(pdk.macro_track_pitch_um, pdk.macro_origin_x_pitch_um,
                        pdk.macro_pdn_x_pitch_um) + 0.001
        elif reference_family == "reference_20261004":
            guard = _latest_reference_snap_guard(workspace, resolved_config)
        elif reference_family in (None, "reference_20260923"):
            from open3dbench.pdks import get_pdk
            pdk = get_pdk(resolved_config["pdk"])
            guard = max(pdk.macro_track_pitch_um, pdk.macro_origin_x_pitch_um,
                        pdk.macro_pdn_x_pitch_um) + 0.001
        else:
            raise ValueError("Unknown placement reference family")
        command += ["--macro-halo-x", str(placement["macro_halo_um"]),
                    "--macro-halo-y", str(placement["macro_halo_um"]),
                    "--macro-snap-guard-um", str(guard)]
    return command, workspace / "Place-3D"


def _deterministic_dreamplace_params(run_dir: Path, output_dir: Path,
                                   *, preserve_reference: bool = False) -> Path:
    """Create a harness-owned deterministic view of frozen placement inputs.

    DREAMPlace's GPU density/electric kernels have an explicit deterministic
    implementation, but the legacy Concurrent LoL wrapper does not set the
    corresponding parameter.  Do not mutate the adopted benchmark input or the
    Open3DBench source tree; publish a small derived parameter file and pass it
    through the existing public ``--base-params`` interface instead.
    """

    source = run_dir / "10_floorplan" / "design_params.json"
    if not source.is_file():
        raise FileNotFoundError(
            "global-placement input is missing frozen design_params.json: {}".format(
                source
            )
        )
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("design_params.json must contain an object")
    if preserve_reference:
        # The frozen native reference omitted this flag (native default 0).
        # Enabling a different kernel here would silently change its recipe.
        if (data.get("deterministic_flag", 0) != 0 or data.get("use_bb", 0) != 0
                or data.get("num_threads") != 4):
            raise ValueError("Frozen reference placement runtime parameters differ")
    else:
        data["deterministic_flag"] = 1
    target = output_dir / "harness_inputs" / "design_params.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _validate_dreamplace_global_output(run_dir: Path) -> Dict[str, str]:
    stage = run_dir / "30_placement" / "stages" / "02_global_place"
    required = {
        "manifest": stage / "stage_manifest.json",
        "params": stage / "params.json",
        "log": stage / "run.log",
        "placement": stage / "placement.gp.def",
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                "DREAMPlace global-placement output {} is missing: {}".format(
                    name, path
                )
            )
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if int(manifest.get("returncode", -1)) != 0:
        raise RuntimeError("DREAMPlace manifest reports a failed placement")
    forbidden = run_dir / "30_placement" / "die_init"
    if forbidden.exists():
        raise RuntimeError(
            "global-placement-only evaluation unexpectedly created die_init"
        )
    odbs = list((run_dir / "30_placement").rglob("*.odb"))
    if odbs:
        raise RuntimeError(
            "global-placement-only evaluation unexpectedly created ODB files"
        )
    return {name: str(path) for name, path in required.items()}


def evaluate_task(
    task_dir: Path,
    candidate_dir: Path,
    open3dbench_root: Path,
    output_dir: Path,
    *,
    baseline_run: Optional[Path] = None,
    reference_run: Optional[Path] = None,
    base_config: Optional[Path] = None,
    dry_run: bool = False,
    keep_workspace: bool = True,
    keep_runtime: bool = True,
    compact_artifacts: bool = False,
    create_routing_reference: bool = False,
    timeout_seconds: int = 21600,
    seed_schedule: Optional[Dict[str, int]] = None,
    execution_phase: Optional[str] = None,
    continuation_sha256: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate one candidate without writing to the Open3DBench source tree."""

    spec = load_task(task_dir)
    from .model import require_executable_task
    require_executable_task(spec)
    candidate_dir = candidate_dir.expanduser().resolve()
    open3dbench_root = open3dbench_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Any] = {
        "task": spec.task_id,
        "candidate": str(candidate_dir),
        "output_dir": str(output_dir),
        "cascade": [],
        "native_flow_execution": [],
    }
    try:
        phase_binding = None
        if execution_phase is not None:
            from .phased import request_binding, load_continuation
            if (spec.task_id != "tp_gnn_post_placement_timing" or dry_run
                    or execution_phase not in {"frontend", "backend"}):
                raise ValueError("Phased local execution requires the maintained TP-GNN timing task")
            phase_binding = request_binding(spec, candidate_dir, open3dbench_root, output_dir,
                baseline_run=baseline_run, reference_run=reference_run, base_config=base_config,
                seed_schedule=seed_schedule, timeout_seconds=timeout_seconds,
                keep_workspace=keep_workspace, keep_runtime=keep_runtime,
                compact_artifacts=compact_artifacts)
            if execution_phase == "backend":
                state = load_continuation(output_dir, continuation_sha256, phase_binding)
                artifacts = state["artifacts"]
                return _execute_prepared(state, phase="backend")
            if continuation_sha256 is not None:
                raise ValueError("Frontend must not consume a continuation")
        elif continuation_sha256 is not None:
            raise ValueError("Continuation requires backend execution")
        if spec.task_id in REFERENCE_PLACEMENT_TASKS and baseline_run is None:
            raise ValueError("Reference-aligned placement requires its frozen baseline run")
        if create_routing_reference:
            if spec.kind not in ROUTING_SOURCE_TASK_KINDS:
                raise ValueError(
                    "--create-routing-reference is only valid for routing "
                    "source-evolution tasks"
                )
            if reference_run is not None:
                raise ValueError(
                    "routing reference creation must not consume another reference run"
                )
            artifacts["routing_reference_creation"] = {
                "status": "enabled",
                "score_semantics": "unscaled bootstrap proxy; raw GRT metrics become the reference",
            }
        candidate = load_candidate(candidate_dir)
        verification = verify_candidate(spec, candidate, open3dbench_root)
        from . import current_nesterov
        current_nesterov_profile = current_nesterov.selected(spec, base_config)
        if seed_schedule is not None:
            from .seeds import cell_contract
            if dry_run:
                raise ValueError("Paired-seed cells require full native evaluation")
            if spec.task_id in REFERENCE_PLACEMENT_TASKS and not current_nesterov_profile:
                raise ValueError("Multi-seed Nesterov requires the current-source profile")
            artifacts["matrix_cell_contract"] = cell_contract(
                spec, candidate, open3dbench_root, baseline_run, reference_run,
                base_config, seed_schedule)
            artifacts["matrix_reference"] = reference_run is None
        from .scoped_reference import TASKS as SCOPED_TASKS, admit_reference
        if spec.task_id in SCOPED_TASKS or spec.kind == "hbt_assignment_policy" or current_nesterov_profile:
            if dry_run:
                artifacts["reference_admission"] = {"mode": "deferred-dry-run"}
            elif current_nesterov_profile:
                artifacts["reference_admission"] = current_nesterov.admit(
                    spec, candidate, open3dbench_root, baseline_run, reference_run, base_config or spec.base_config,
                    seed_schedule=seed_schedule)
                artifacts["reference_profile"] = current_nesterov.PROFILE
            else:
                artifacts["reference_admission"] = admit_reference(
                    spec, candidate, open3dbench_root, baseline_run, reference_run)
            snapshot = output_dir / "candidate_snapshot.json"
            with snapshot.open("x", encoding="utf-8") as stream:
                json.dump(candidate, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        embedded_patch = candidate.get("source_patch", "")
        patch_path = candidate_dir / "changes.patch"
        if embedded_patch:
            patch_info = verification["patch"]
            patch_path = output_dir / "candidate.patch"
            patch_path.write_text(embedded_patch, encoding="utf-8")
        else:
            patch_info = validate_patch_scope(spec, patch_path)
        artifacts["patch"] = patch_info
        artifacts["cascade"].append({"name": "policy", "status": "passed"})

        resolved_config_path = output_dir / "resolved_config.yaml"
        resolved_config = _write_resolved_config(
            spec,
            candidate_dir,
            resolved_config_path,
            candidate,
            base_config=base_config, seed_schedule=seed_schedule,
        )
        if seed_schedule is not None and spec.task_id in {
                "tp_gnn_post_placement_timing", "tp_gnn_binary_loss_timing",
                "tp_gnn_loss_decoder_timing"}:
            _bind_full_flow_method_seeds(resolved_config, seed_schedule, output_dir, open3dbench_root)
            resolved_config_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
        if spec.kind == "hbt_assignment_policy":
            from .hbt_native import validate_recipe, validate_input_identity
            validate_recipe(resolved_config, open3dbench_root)
            if not 0 < timeout_seconds <= 21600:
                raise ValueError("HBT policy time budget must be within six hours")
            if not dry_run:
                if baseline_run is None:
                    raise ValueError("HBT policy requires an immutable five-stage prefix")
                artifacts["hbt_input_prefix"] = validate_input_identity(baseline_run, resolved_config)
        if spec.kind == "tp_gnn_post_placement_timing":
            from .post_placement_timing import validate_recipe
            validate_recipe(resolved_config, open3dbench_root, expected_decoder="global-2way")
        if spec.kind in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
            from .gnn_evolution_timing import validate_recipe as validate_evolution_recipe
            validate_evolution_recipe(resolved_config, open3dbench_root, task_id=spec.task_id)
        from . import reference_latest
        latest_profile = reference_latest.is_latest_config(base_config or spec.base_config)
        if latest_profile:
            initial_config = yaml.safe_load((base_config or spec.base_config).read_text())
            reference_latest.validate_reference_config(initial_config, initial_config["design"])
            if not dry_run:
                runtime_image = os.environ.get("OPEN3DBENCH_IMAGE_ID")
                if spec.kind in ROUTING_SOURCE_TASK_KINDS and runtime_image != reference_latest.IMAGE_ID:
                    # The subsequent prepare_openroad_runtime check authenticates
                    # the image-owned binary and complete patched source tree.
                    # Here bind this candidate image to its exact base/source/task.
                    receipt_path = os.environ.get("OPEN3D_LOCAL_OPENROAD_PREBUILT_RECEIPT")
                    if not receipt_path:
                        raise ValueError("Latest routing candidate requires its installed-runtime receipt")
                    receipt = reference_latest.read_json(receipt_path)
                    from .runtime_equivalence import validate_latest_build_base
                    validate_latest_build_base(receipt)
                    candidate_digest = hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
                    if (receipt.get("kind") != "routing-source-installed-openroad"
                            or receipt.get("image_id") != runtime_image
                            or receipt.get("source_commit") != reference_latest.SOURCE_REVISION
                            or receipt.get("openroad_commit") != reference_latest.NATIVE_REVISION
                            or receipt.get("task") != spec.task_id or receipt.get("candidate_sha256") != candidate_digest
                            or not any(name.startswith("OpenROAD-3D/openroad/") for name in patch_info["paths"])):
                        raise ValueError("Latest routing candidate runtime contract differs")
                else:
                    reference_latest.validate_reference_image(runtime_image)
            artifacts["reference_profile"] = reference_latest.BASELINE_ID
        artifacts["resolved_config"] = str(resolved_config_path)
        artifacts["base_config"] = str(
            (base_config or spec.base_config).expanduser().resolve()
        )
        if spec.frozen_inputs and not dry_run:
            artifacts["frozen_inputs"] = _materialize_frozen_inputs(
                spec,
                resolved_config,
                resolved_config_path,
                output_dir,
                baseline_run=baseline_run,
                reference_run=reference_run,
            )
            artifacts["cascade"].append(
                {"name": "frozen_inputs", "status": "passed"}
            )
        elif spec.frozen_inputs:
            artifacts["frozen_inputs"] = {"status": "deferred-dry-run"}
        if spec.kind in ROUTING_SOURCE_TASK_KINDS and not dry_run:
            if create_routing_reference:
                artifacts["cascade"].append(
                    {"name": "reference", "status": "bootstrap"}
                )
            else:
                artifacts["reference_validation"] = _validate_routing_source_reference(
                    spec,
                    baseline_run,
                    reference_run,
                    resolved_config,
                    source_root=open3dbench_root,
                )
                artifacts["cascade"].append(
                    {"name": "reference", "status": "passed"}
                )
        if (not dry_run and baseline_run is not None
                and spec.kind in PLACEMENT_TASK_KINDS):
            artifacts["placement_input_validation"] = _validate_placement_partition_reference(
                spec, baseline_run, base_config,
            )
            if spec.task_id in REFERENCE_PLACEMENT_TASKS and not current_nesterov_profile:
                from .reference_latest import baseline_for_config
                initial_config = yaml.safe_load((base_config or spec.base_config).read_text())
                selected_baseline = baseline_for_config(initial_config)
                case = initial_config.get("design") if isinstance(initial_config, dict) else None
                if not isinstance(case, str) or not case:
                    raise ValueError("Reference placement requires a design in its trusted base config")
                receipt = selected_baseline.validate_reference_config(initial_config, case)
                receipt["case"] = case
                selected_baseline.validate_reference_image(os.environ.get("OPEN3DBENCH_IMAGE_ID"))
                receipt["physical_input_sha256"] = selected_baseline.validate_physical_prefix(
                    baseline_run, case, fixed_partition=True)
                artifacts["reference_alignment"] = receipt
        if spec.kind in STA_TWO_PASS_TASK_KINDS and not dry_run:
            artifacts["reference_validation"] = _validate_sta_codesign_reference(
                spec, baseline_run, reference_run
            )
        if spec.kind in THERMAL_TASK_KINDS and not dry_run and reference_run is not None:
            reference_metrics = extract_metrics(
                spec.kind, reference_run, None, frozen_input_run=baseline_run)
            artifacts["reference_validation"] = {
                "status": "passed",
                "global_average_temperature_k": reference_metrics[
                    "global_average_temperature_k"
                ],
                "thermal_model": reference_metrics["thermal_model"],
            }
            artifacts["cascade"].append(
                {
                    "name": "reference",
                    "status": artifacts["reference_validation"]["status"],
                }
            )

        if latest_profile:
            resolved_config = reference_latest.bind_recipe_paths(resolved_config, open3dbench_root)
            resolved_config_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
        open3dbench_python = os.environ.get(
            "OPEN3DBENCH_PYTHON", os.environ.get("PYTHON", "python3")
        )
        plan_command = [
            open3dbench_python,
            "-m",
            "open3dbench",
            "--repo",
            str(open3dbench_root),
            "plan",
            str(resolved_config_path),
        ]
        # Keep planning independent of the caller's working directory and of
        # any site-specific Python startup behavior.  The requested checkout
        # is the authoritative implementation for this validation step.
        plan_environment = dict(os.environ)
        inherited_pythonpath = [
            item
            for item in plan_environment.get("PYTHONPATH", "").split(os.pathsep)
            if item and item != str(open3dbench_root)
        ]
        plan_environment["PYTHONPATH"] = os.pathsep.join(
            [str(open3dbench_root)] + inherited_pythonpath
        )
        plan = subprocess.run(
            plan_command,
            cwd=str(open3dbench_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
            env=plan_environment,
        )
        plan_log = output_dir / "plan.log"
        plan_log.write_text(plan.stdout, encoding="utf-8")
        artifacts["plan_log"] = str(plan_log)
        if plan.returncode:
            raise RuntimeError("Open3DBench rejected the candidate configuration; see plan.log")
        artifacts["cascade"].append({"name": "interface", "status": "passed"})

        run_dir = output_dir / "run"
        workspace = output_dir / "workspace" / "Open3DBench-V2.0"
        replay_start = _replay_start_stage(spec, resolved_config, plan.stdout)
        artifacts["replay_start_stage"] = replay_start
        command = [
            open3dbench_python,
            "-m",
            "open3dbench",
            "--repo",
            str(workspace),
            "run",
            str(resolved_config_path),
            "--from",
            replay_start,
            "--to",
            spec.stop_stage,
            "--resume",
            "--run-dir",
            str(run_dir),
        ]
        from local_tasks.routing_source_common import standalone_grt_supported
        if spec.kind in ROUTING_SOURCE_TASK_KINDS and standalone_grt_supported(open3dbench_root):
            # Standalone GRT has no DRT side effect and does not mark root routing complete.
            command = [open3dbench_python, "-m", "open3dbench", "--repo", str(workspace),
                       "global-route", str(resolved_config_path), "--run-dir", str(run_dir)]
            artifacts["replay_start_stage"] = "global-route"
        command_cwd = workspace
        artifacts["command"] = command
        artifacts["workspace"] = str(workspace)
        artifacts["run_dir"] = str(run_dir)
        artifacts["stage_log"] = str(output_dir / "stage.log")
        artifacts["evaluation_result"] = str(output_dir / "evaluation_result.json")
        artifacts["evaluation_boundary"] = spec.evaluation_boundary
        if dry_run:
            artifacts["cascade"].append({"name": "stage", "status": "dry_run"})
            result = EvaluationResult(
                metrics={"combined_score": 0.0, "runs_successfully": 0.0, "dry_run": 1.0},
                artifacts=artifacts,
            )
            (output_dir / "evaluation_result.json").write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return result

        if spec.upstream_stages and baseline_run is None:
            raise ValueError("task {} requires --baseline-run".format(spec.task_id))

        mutable_submodules = (
            ["OpenROAD-3D/openroad"]
            if any(
                path.startswith("OpenROAD-3D/openroad/")
                for path in patch_info.get("paths", [])
            )
            else []
        )
        isolated_clone(
            open3dbench_root,
            workspace,
            mutable_submodules=mutable_submodules,
        )
        evaluator_patch = select_evaluator_source_patch(spec.kind, workspace)
        if evaluator_patch is not None:
            if not evaluator_patch.is_file():
                raise FileNotFoundError(
                    "trusted evaluator source patch is missing: {}".format(
                        evaluator_patch
                    )
                )
            artifacts["evaluator_source_patch"] = {
                "path": str(evaluator_patch),
                "sha256": _file_sha256(evaluator_patch),
                **apply_candidate_patch(workspace, evaluator_patch),
            }
        dreamplace_source = materialize_dreamplace_source_tree(
            open3dbench_root,
            workspace,
            list(patch_info.get("paths", [])),
        )
        if dreamplace_source is not None:
            artifacts["dreamplace_source"] = str(dreamplace_source)
        apply_candidate_patch(workspace, patch_path)
        source_validation = validate_candidate_workspace(
            workspace,
            list(patch_info.get("paths", [])),
            open3dbench_python,
        )
        artifacts["source_validation"] = source_validation
        artifacts["cascade"].append({"name": "source_validation", "status": "passed"})
        run_environment = dict(os.environ)
        if spec.kind in STA_TWO_PASS_TASK_KINDS:
            run_environment["OPEN3D_OPT_PASSES"] = "2"
            artifacts["opt_passes"] = 2
        if spec.kind in ROUTING_SOURCE_TASK_KINDS:
            # These tasks target the native `global_routing` stage, which
            # publishes an auditable checkpoint before DRT/antenna/DRC.
            artifacts["routing_evaluation_boundary"] = "global-route-only"
        dreamplace_overlay = prepare_dreamplace_overlay(
            open3dbench_root,
            workspace,
            output_dir,
            list(patch_info.get("paths", [])),
        )
        if dreamplace_overlay is not None:
            run_environment["PLACE3D_DREAMPLACE_INSTALL"] = str(dreamplace_overlay)
            artifacts["dreamplace_runtime"] = str(dreamplace_overlay)
        openroad_runtime = prepare_openroad_runtime(
            workspace,
            output_dir,
            list(patch_info.get("paths", [])),
        )
        if openroad_runtime is not None:
            run_environment["OPENROAD_EXE"] = str(openroad_runtime)
            run_environment = _openroad_runtime_environment(run_environment)
            artifacts["openroad_runtime"] = str(openroad_runtime)
            artifacts["openroad_build_log"] = str(output_dir / "openroad-build.log")
        if spec.upstream_stages:
            assert baseline_run is not None
            if spec.kind == "hbt_assignment_policy":
                from .hbt_native import copy_prefix
                artifacts["reference_import"] = copy_prefix(baseline_run, run_dir)
            elif (spec.kind in ROUTING_SOURCE_TASK_KINDS
                  and standalone_grt_supported(open3dbench_root)):
                artifacts["reference_import"] = reference_latest.copy_grt_prefix(
                    baseline_run, run_dir, spec.upstream_stages)
            else:
                copy_upstream_stages(baseline_run.resolve(), run_dir, spec.upstream_stages)
            artifacts["upstream_materialization"] = (
                "hardlink-large-frozen-files"
                if os.environ.get("OPEN3D_LINK_FROZEN_UPSTREAM") == "1"
                else "copy"
            )
        if spec.evaluation_boundary == "dreamplace-global-placement":
            deterministic_params = _deterministic_dreamplace_params(
                run_dir, output_dir,
                preserve_reference=(spec.task_id in REFERENCE_PLACEMENT_TASKS | COMPONENT_PLACEMENT_TASKS
                                    and not current_nesterov_profile),
            )
            command, command_cwd = _dreamplace_global_placement_command(
                spec,
                resolved_config,
                workspace,
                run_dir,
                open3dbench_python,
                base_params_override=deterministic_params,
                reference_family=reference_latest.BASELINE_ID if latest_profile else None,
            )
            artifacts["command"] = command
            artifacts["database_materialization"] = "disabled"
            artifacts["deterministic_placement"] = (current_nesterov_profile or
                spec.task_id not in REFERENCE_PLACEMENT_TASKS | COMPONENT_PLACEMENT_TASKS)
            if spec.task_id in REFERENCE_PLACEMENT_TASKS:
                artifacts["placement_kernel_policy"] = (
                    "current-deterministic" if current_nesterov_profile else "preserve-native-reference-default")
            artifacts["deterministic_base_params"] = str(deterministic_params)
        return _execute_prepared({
            "spec": spec,
            "candidate": candidate,
            "workspace": workspace,
            "run_dir": run_dir,
            "resolved_config": resolved_config,
            "open3dbench_python": open3dbench_python,
            "run_environment": run_environment,
            "artifacts": artifacts,
            "command": command,
            "command_cwd": command_cwd,
            "output_dir": output_dir,
            "reference_run": reference_run,
            "baseline_run": baseline_run,
            "timeout_seconds": timeout_seconds,
            "seed_schedule": seed_schedule,
            "current_nesterov_profile": current_nesterov_profile,
            "dreamplace_overlay": dreamplace_overlay,
            "keep_workspace": keep_workspace,
            "keep_runtime": keep_runtime,
            "compact_artifacts": compact_artifacts,
            "phase_binding": phase_binding,
        }, phase=execution_phase)
    except Exception as error:
        if compact_artifacts:
            _discard_heavy_evaluation_artifacts(output_dir, artifacts)
        else:
            if not keep_workspace:
                shutil.rmtree(output_dir / "workspace", ignore_errors=True)
            if not keep_runtime:
                shutil.rmtree(output_dir / "runtime", ignore_errors=True)
        result = _failure(error, artifacts)
        (output_dir / "evaluation_result.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result



def _bind_full_flow_method_seeds(config, seeds, output, source):
    """Use the global flow's existing native JSON seed adapter for local TP-GNN."""
    from grt_bench.flow import bind_method_seeds
    for stage in ("partition", "placement"):
        path = Path(config[stage]["config"])
        if not path.is_absolute():
            path = Path(source) / "Place-3D" / path
        config[stage]["config"] = str(path.resolve(strict=True))
    bind_method_seeds(config, seeds, Path(output) / "method_seeds")


def _execute_prepared(state, *, phase=None):
    """Execute/measure one prepared cell, or its authenticated CPU continuation."""
    spec = state["spec"]
    candidate = state["candidate"]
    workspace = state["workspace"]
    run_dir = state["run_dir"]
    resolved_config = state["resolved_config"]
    open3dbench_python = state["open3dbench_python"]
    run_environment = state["run_environment"]
    artifacts = state["artifacts"]
    command = state["command"]
    command_cwd = state["command_cwd"]
    output_dir = state["output_dir"]
    reference_run = state["reference_run"]
    baseline_run = state["baseline_run"]
    timeout_seconds = state["timeout_seconds"]
    seed_schedule = state["seed_schedule"]
    current_nesterov_profile = state["current_nesterov_profile"]
    dreamplace_overlay = state["dreamplace_overlay"]
    keep_workspace = state["keep_workspace"]
    keep_runtime = state["keep_runtime"]
    compact_artifacts = state["compact_artifacts"]
    phase_binding = state["phase_binding"]
    from .phased import split_command, save_continuation
    elapsed = state.get("frontend_runtime_seconds", 0.0)
    remaining_timeout = timeout_seconds - elapsed
    if remaining_timeout <= 0:
        raise TimeoutError("Local evaluation budget exhausted before CPU continuation")
    if phase is not None:
        command = split_command(command, phase)
    started = time.monotonic() - elapsed
    log_name = "backend-stage.log" if phase == "backend" else "stage.log"
    with (output_dir / log_name).open("w", encoding="utf-8") as log:
        proc = _run_native_flow(
            command,
            invocation="stage",
            artifacts=artifacts,
            cwd=str(command_cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=remaining_timeout,
            env=run_environment,
        )
    artifacts["runtime_seconds"] = time.monotonic() - started
    artifacts["exit_code"] = proc.returncode
    if proc.returncode:
        raise RuntimeError("Open3DBench stage failed; see stage.log")
    if phase == "frontend":
        if seed_schedule is not None:
            from .seeds import validate_executed_seeds
            artifacts["executed_seed_evidence"] = validate_executed_seeds(spec, run_dir, seed_schedule)
        state["frontend_runtime_seconds"] = time.monotonic() - started
        checkpoint = save_continuation(state)
        return EvaluationResult(metrics={"runs_successfully": 0.0}, artifacts={
            "phase": "frontend-complete", "continuation": str(checkpoint),
            "evaluation_pending": True})
    if spec.kind in {"tp_gnn_bin_timing", "tp_gnn_post_placement_timing", "hbt_assignment_policy"}:
        from .post_placement_timing import snapshot_command
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("evaluation budget exhausted before post-placement STA")
        post_command = snapshot_command(workspace, run_dir, resolved_config,
                                        open3dbench_python, run_environment)
        artifacts["sta_command"] = post_command
        artifacts["opt_passes"] = 0
        with (output_dir / "post-placement-sta.log").open("w", encoding="utf-8") as log:
            post = _run_native_flow(post_command, invocation="post_placement_sta", artifacts=artifacts,
                cwd=str(workspace), stdout=log, stderr=subprocess.STDOUT,
                check=False, timeout=remaining, env=run_environment)
        artifacts["exit_code"] = post.returncode
        artifacts["runtime_seconds"] = time.monotonic() - started
        if post.returncode:
            raise RuntimeError("Post-placement STA failed; see post-placement-sta.log")
    if spec.kind == "tp_gnn_placement_cts_thermal":
        from .cts_thermal import snapshot_command

        post_command = snapshot_command(
            workspace, run_dir, resolved_config, open3dbench_python, run_environment
        )
        artifacts["thermal_command"] = post_command
        artifacts["thermal_evaluation_boundary"] = "two-opt-cts-cell-power-v1"
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("evaluation budget exhausted before CTS power/thermal")
        with (output_dir / "thermal.log").open("w", encoding="utf-8") as log:
            post = _run_native_flow(
                post_command, invocation="cts_power_thermal", artifacts=artifacts,
                cwd=str(workspace), stdout=log, stderr=subprocess.STDOUT,
                check=False, timeout=remaining, env=run_environment,
            )
        artifacts["exit_code"] = post.returncode
        artifacts["runtime_seconds"] = time.monotonic() - started
        if post.returncode:
            raise RuntimeError("CTS power/thermal failed; see thermal.log")
    if spec.evaluation_boundary == "dreamplace-global-placement":
        artifacts["global_placement_outputs"] = (
            _validate_dreamplace_global_output(run_dir)
        )
        if spec.task_id in REFERENCE_PLACEMENT_TASKS:
            from . import current_nesterov
            from .reference_placement_evidence import validate_reference_placement_evidence
            artifacts["reference_placement_evidence"] = validate_reference_placement_evidence(
                run_dir, spec.task_id, runtime_overlay=dreamplace_overlay,
                **({"profile": current_nesterov.PROFILE,
                    "placement_seed": seed_schedule["placement"] if seed_schedule else 1000}
                   if current_nesterov_profile else {}))
    if seed_schedule is not None:
        from .seeds import validate_executed_seeds
        artifacts["executed_seed_evidence"] = validate_executed_seeds(spec, run_dir, seed_schedule)
    artifacts["cascade"].append({"name": "stage", "status": "passed"})
    metrics = extract_metrics(spec.kind, run_dir, reference_run.resolve() if reference_run else None,
                              frozen_input_run=baseline_run, task_id=spec.task_id,
                              **({"scoring": spec.scoring} if spec.scoring else {}))
    if spec.kind == "hbt_assignment_policy":
        actual_policy = json.loads((run_dir / "50_hbt/policy.json").read_text())
        if actual_policy != candidate["candidate_files"]["hbt_policy.json"]:
            raise ValueError("Executed HBT policy differs from the submitted candidate")
    metrics["runs_successfully"] = 1.0
    artifacts["cascade"].append({"name": "metrics", "status": "passed"})
    if compact_artifacts:
        _discard_heavy_evaluation_artifacts(output_dir, artifacts)
    else:
        if not keep_workspace:
            shutil.rmtree(output_dir / "workspace", ignore_errors=True)
        if not keep_runtime:
            shutil.rmtree(output_dir / "runtime", ignore_errors=True)
    result = EvaluationResult(metrics=metrics, artifacts=artifacts)
    (output_dir / "evaluation_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result

def _discard_heavy_evaluation_artifacts(
    output_dir: Path, artifacts: Dict[str, Any]
) -> None:
    """Retain auditable scalar/log outputs while removing reproducible bulk data.

    Candidate source, resolved configuration, plan/stage/build logs and the
    evaluation result live directly under ``output_dir`` and remain available
    to the search agent.  The isolated checkout, compiled runtime and copied
    stage tree can each be regenerated from the frozen input and candidate.
    """

    retained_dir = output_dir / "retained_artifacts"
    global_outputs = artifacts.get("global_placement_outputs")
    retained_evidence: Dict[str, str] = {}
    run_dir = output_dir / "run"
    # Preserve bounded algorithm evidence before removing physical databases.
    # This supports feedback audits without retaining duplicate ODB/DEF/SPEF.
    evidence_budget = 32 * 1024 * 1024
    algorithm_evidence = []
    for source in sorted(run_dir.rglob("*")) if run_dir.exists() else ():
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(run_dir)
        if relative.parts[0] not in {"20_partition", "30_placement", "40_pdn", "50_hbt", "55_post_placement_sta", "80_route"}:
            continue
        structured = source.name in {"tp_gnn_report.json", "partition_manifest.json", "params.json",
            "metrics.json", "grt_metrics.json", "stage_manifest.json", "assignment_manifest.json"}
        if not structured and source.suffix not in {".log", ".rpt"}:
            continue
        size = source.stat().st_size
        length = size if structured else min(size, 2 * 1024 * 1024)
        if length > min(evidence_budget, 4 * 1024 * 1024):
            algorithm_evidence.append({"path": str(relative), "bytes": size, "retained": False})
            continue
        destination = retained_dir / "algorithm" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as stream:
            stream.seek(size - length)
            destination.write_bytes(stream.read(length))
        evidence_budget -= length
        algorithm_evidence.append({"path": str(relative), "bytes": size, "retained": True,
            "retained_bytes": length, "offset": size - length,
            "source_sha256": _file_sha256(source), "retained_sha256": _file_sha256(destination)})
    artifacts["algorithm_evidence"] = {"directory": str(retained_dir / "algorithm"),
                                       "files": algorithm_evidence}
    for source, filename, key in (
        (run_dir / "30_placement" / "metrics.json", "placement_metrics.json", "placement_metrics"),
        (run_dir / "60_opt" / "metrics.json", "two_pass_opt_metrics.json", "two_pass_opt_metrics"),
        (run_dir / "60_opt" / "opt.log", "two_pass_opt.log", "two_pass_opt_log"),
        (run_dir / "95_thermal" / "metrics.json", "thermal_metrics.json", "thermal_metrics"),
        (run_dir / "95_thermal" / "hotspot.log", "hotspot.log", "hotspot_log"),
        (run_dir / "70_cts" / "metrics.json", "cts_metrics.json", "cts_metrics"),
        (run_dir / "95_thermal_snapshot" / "metrics.json", "cts_thermal_metrics.json", "cts_thermal_metrics"),
        (run_dir / "95_thermal_snapshot" / "snapshot_manifest.json", "cts_thermal_manifest.json", "cts_thermal_manifest"),
        (run_dir / "95_thermal_snapshot" / "hotspot.log", "cts_hotspot.log", "cts_hotspot_log"),
        (run_dir / "run.json", "run.json", "run_manifest"),
    ):
        if source.is_file():
            retained_dir.mkdir(parents=True, exist_ok=True)
            destination = retained_dir / filename
            shutil.copy2(source, destination)
            retained_evidence[key] = str(destination)
    if isinstance(global_outputs, dict):
        retained_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in (
            ("params", "placement_params.json"),
            ("manifest", "placement_stage_manifest.json"),
        ):
            source_value = global_outputs.get(key)
            source = Path(source_value) if isinstance(source_value, str) else None
            if source is not None and source.is_file():
                destination = retained_dir / filename
                shutil.copy2(source, destination)
                retained_evidence[key] = str(destination)
                global_outputs[key] = str(destination)

        log_value = global_outputs.get("log")
        log_source = Path(log_value) if isinstance(log_value, str) else None
        if log_source is not None and log_source.is_file():
            log_destination = retained_dir / "placement_run_tail.log"
            max_tail_bytes = 512 * 1024
            with log_source.open("rb") as source_stream:
                source_stream.seek(0, 2)
                source_size = source_stream.tell()
                source_stream.seek(max(0, source_size - max_tail_bytes))
                log_destination.write_bytes(source_stream.read())
            retained_evidence["log_tail"] = str(log_destination)
            global_outputs["log"] = str(log_destination)

    removed = []
    for name in ("workspace", "runtime", "run"):
        path = output_dir / name
        if path.is_symlink():
            # An opt-in evaluation scratch layout may point workspace/runtime
            # at a local SSD while the durable result directory lives on NAS.
            # Compact retention owns the link, not its target; the harness
            # removes the scratch tree after evaluate_task returns.
            path.unlink()
            removed.append(name)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                removed.append(name)
    artifacts["artifact_retention"] = {
        "mode": "compact",
        "removed_reproducible_directories": removed,
        "retained": [
            "candidate",
            "resolved_config",
            "plan_log",
            "stage_log",
            "openroad_build_log",
            "evaluation_result",
            "placement_params",
            "placement_stage_manifest",
            "placement_run_tail",
            "two_pass_opt_metrics",
            "two_pass_opt_log",
            "thermal_metrics",
            "hotspot_log",
            "run_manifest",
        ],
        "retained_evidence": retained_evidence,
    }
