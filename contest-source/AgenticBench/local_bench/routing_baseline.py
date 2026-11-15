"""Read-only qualification of new TP-GNN/Concurrent GRT baselines.

This launcher gate complements the common evaluator's case/budget binding.
Historical profiles remain readable by the generic evaluator, but cannot be
silently selected as the new routing baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import yaml

from .benchmark_input import _validate_stage_manifest, load_benchmark_input_profile, validate_benchmark_input
from .evaluation import _validate_routing_source_reference
from .model import load_task
from .metrics import routing_net_ordering_metrics
from .tp_gnn_contract import TP_GNN_METHODS, validate_tp_gnn_method


TASKS = (
    "global_routing_ripup_policy",
    "global_routing_maze_cost", "global_routing_iteration_controller",
)


def _routing_stage(reference: Path) -> Path:
    global_route = reference / "75_global_route"
    return global_route if global_route.is_dir() else reference / "80_route"


FRAMEWORK = Path(__file__).resolve().parents[1]


def routing_profile(design: str):
    if not design or Path(design).name != design or design in (".", ".."):
        raise ValueError("design must be a registered case name")
    return load_benchmark_input_profile(design + "_lol_ng45_tp_gnn_cts")


def _read(path: Path) -> Dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(stage: Path, manifest: Dict[str, Any], key: str) -> Path:
    record = manifest.get("artifacts", {}).get(key)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("missing hashed {} artifact in {}".format(key, stage))
    path = (stage / record["path"]).resolve()
    path.relative_to(stage.resolve())
    if not path.is_file() or path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
        raise ValueError("artifact integrity mismatch: {}".format(path))
    return path


def _validate_reference_upstream(baseline: Path, reference: Path, directories) -> None:
    # The common copier only rebases these three run-owned input paths.
    # All other content, including ODB/DEF/SDC/method evidence, must be identical.
    for directory in directories:
        stage = reference / directory
        manifest = _read(stage / "stage_manifest.json")
        for key, record in manifest["artifacts"].items():
            if record["path"] != "design_params.json":
                _artifact(stage, manifest, key)
                continue
            frozen = _read(baseline / directory / "design_params.json")
            copied = _read(stage / "design_params.json")
            if copied == frozen:
                continue
            expected = dict(frozen)
            for parameter, filename in (
                ("def_input", "design.def"), ("verilog_input", "design.v"), ("sdc_input", "design.sdc"),
            ):
                if (stage / filename).is_file():
                    expected[parameter] = str((stage / filename).resolve())
            if copied != expected:
                raise ValueError("reference changed frozen design parameters: {}".format(stage))


def validate_routing_baseline(benchmark_input: Path, reference_run: Path, design: str,
                              task: str = "global_routing_ripup_policy") -> Dict[str, Any]:
    if task not in TASKS:
        raise ValueError("unsupported routing source task: {}".format(task))
    profile = routing_profile(design)
    spec = load_task(FRAMEWORK / "local_tasks" / task)
    bundle = validate_benchmark_input(benchmark_input, task_id=spec.task_id)
    if bundle["benchmark_id"] != profile.profile_id:
        raise ValueError("new routing baseline requires {}; regenerate, do not relabel historical inputs".format(profile.profile_id))
    config = yaml.safe_load(profile.config.read_text(encoding="utf-8"))
    if (config.get("design") != design or config.get("type") != "lol" or config.get("pdk") != "ng45"
            or config.get("partition", {}).get("algorithm") not in TP_GNN_METHODS
            or config.get("placement", {}).get("algorithm") != "concurrent-dreamplace"):
        raise ValueError("routing profile must select TP-GNN and Concurrent DREAMPlace on LoL/ng45")
    baseline, reference = Path(bundle["run_dir"]), reference_run.expanduser().resolve()
    manifests = {}
    for stage, directory in profile.required_stages.items():
        manifest = _read(baseline / directory / "stage_manifest.json")
        if any(manifest.get(key) != value for key, value in
               (("design", design), ("integration", "lol"), ("pdk", "ng45"))):
            raise ValueError("frozen {} stage belongs to another case/platform".format(stage))
        manifests[stage] = manifest
    if manifests["placement"].get("method") != "concurrent-dreamplace":
        raise ValueError("recorded placement method is not Concurrent DREAMPlace")
    partition = _read(_artifact(baseline / "20_partition", manifests["partition"], "method_manifest"))
    report = partition.get("method_report", {})
    method = validate_tp_gnn_method(
        report, partition.get("method"), manifests["partition"].get("method"),
        expected_method=config["partition"]["algorithm"],
    )
    observed = report.get("config", {})
    if (observed.get("decoder") != "global-2way"
            or observed.get("timing_driven") is not True
            or observed.get("feature_regime") != "partition-last"
            or observed.get("timing_mode") != "partition-last"):
        raise ValueError("recorded TP-GNN reference must use global-2way and timing-driven partition-last")
    timing = _read(_artifact(baseline / "20_partition", manifests["partition"], "timing_manifest"))
    if timing.get("timing_status") != "full_coverage_partition_last":
        raise ValueError("TP-GNN reference is missing full-coverage timing features")
    checkpoint = _read(_artifact(baseline / "70_cts", manifests["cts"], "checkpoint"))
    if (checkpoint.get("stage") != "cts" or checkpoint.get("ready_for_route") is not True
            or set(checkpoint.get("dies", {})) != {"bottom", "top"}):
        raise ValueError("CTS checkpoint is not ready for two-die routing")
    bound = _validate_routing_source_reference(spec, baseline, reference, config)
    _validate_reference_upstream(baseline, reference, profile.required_stages.values())
    route_stage = _routing_stage(reference)
    expected_stage = (
        "global_routing" if route_stage.name == "75_global_route" else "routing"
    )
    route_manifest = _validate_stage_manifest(
        route_stage / "stage_manifest.json", expected_stage
    )
    metrics_path = _artifact(route_stage, route_manifest, "metrics")
    if metrics_path != (route_stage / "metrics.json").resolve():
        raise ValueError("reference metrics must be the canonical hashed routing artifact")
    metrics = _read(metrics_path)
    for die, row in metrics["dies"].items():
        if row.get("status") != "success":
            raise ValueError("reference {} GRT did not succeed".format(die))
        for key in ("grt_wirelength_um", "grt_total_overflow"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid reference {} {}".format(die, key))
        if row.get("ndr_policy") != "preserve" or row.get("layer_window") != ["metal2", "metal10"]:
            raise ValueError("reference die routing layer/NDR policy mismatch")
    if metrics.get("ready_for_extract") is not False or metrics.get("unexpected_warnings") != 0:
        raise ValueError("reference must be warning-audited GRT-only, not a routed/signoff result")
    native = routing_net_ordering_metrics(reference, None)
    if native["net_order_diagnostics_present"] != 1 or native["net_order_changed_calls"] != 0:
        raise ValueError("reference must have reachable, behavior-preserving native net-order diagnostics")
    return {
        "status": "passed", "benchmark_id": profile.profile_id,
        "partition": method, "decoder": "global-2way", "placement": "concurrent-dreamplace",
        "benchmark_input": bundle["manifest"], "reference_run": str(reference),
        "evaluation_boundary": "global-route-only", "reference_binding": bound,
    }


def validate_initial_replay(result_path: Path, reference: Path) -> None:
    """Do not spend a search budget until the initial policy reproduces reference."""
    result = _read(result_path).get("metrics", {})
    raw = _read(_routing_stage(reference) / "metrics.json")
    expected = {key: sum(row[key] for row in raw["dies"].values())
                for key in ("grt_total_overflow", "grt_wirelength_um")}
    for key, value in {"runs_successfully": 1, "routing_complete": 1, "combined_score": 1, **expected}.items():
        observed = result.get(key)
        if (isinstance(observed, bool) or not isinstance(observed, (int, float))
                or not math.isfinite(observed) or not math.isclose(observed, value, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("initial policy did not reproduce reference: {}".format(key))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--benchmark-input", required=True, type=Path)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--identity-result", type=Path, help="Also qualify an evaluated initial policy (never calls EDA or an API).")
    args = parser.parse_args(argv)
    report = validate_routing_baseline(args.benchmark_input, args.reference_run, args.case, args.task)
    if args.identity_result:
        validate_initial_replay(args.identity_result, args.reference_run)
        report["initial_policy_identity"] = "passed"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
