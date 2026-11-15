"""Complete, paired local-task matrices over the existing single-cell evaluator.

The matrix never changes a task's physical boundary. Contest GNN tasks apply
their shared timing denominators to verified native measurements here. Each
candidate spends one matrix budget; every case has equal weight, and every seed
has equal weight within its case. Failed cells cannot disappear from the mean.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import statistics

import yaml

from grt_bench.common import digest, finite, read, sha, write
from grt_bench.contract import identifier
from .model import EvaluationResult, ROUTING_SOURCE_TASK_KINDS, load_task
from .seeds import validate_schedule


def _require_reference_activity(spec, metrics):
    """A process-success receipt cannot qualify an unexercised routing hook."""
    if spec.kind in ROUTING_SOURCE_TASK_KINDS and metrics.get("grt_complete") != 1:
        raise ValueError("Unqualified routing reference: required policy hook was not exercised; "
                         "choose an input/configuration with native hook evidence before evolution")


class LocalEvaluationMatrix:
    def __init__(self, path, spec, *, references_required=True):
        self.path = Path(path).expanduser().resolve(strict=True)
        value = yaml.safe_load(self.path.read_text())
        required = {"schema_version", "task", "cases", "seeds", "cells"}
        if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                or value["schema_version"] not in (1, 2)
                or set(value) != (required if value["schema_version"] == 1 else required | {"case_seeds"})):
            raise ValueError("Local matrix requires schema_version=1 or explicit schema_version=2 case_seeds")
        if value["task"] != spec.task_id:
            raise ValueError("Matrix belongs to a different local task")
        from .task_inventory import MATRIX_TASK_DIRECTORIES
        supported = {load_task(spec.root.parent / name).task_id for name in MATRIX_TASK_DIRECTORIES}
        # Contest 2 supplies its own three-case matrix and two baseline-specific
        # references; the historical local matrix profile remains unchanged.
        supported |= {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}
        if spec.task_id not in supported:
            raise ValueError("Task has no defined seed-matrix contract; matrix execution supports the original eight local tasks")
        cases, seeds, cells = value["cases"], value["seeds"], value["cells"]
        if not isinstance(cells, list) or not cells:
            raise ValueError("Matrix cells must be a nonempty list")
        if not isinstance(cases, list) or not cases or not isinstance(seeds, list) or not seeds:
            raise ValueError("Empty case/seed matrix")
        if len({identifier(c) for c in cases}) != len(cases):
            raise ValueError("Duplicate case")
        vectors = {}
        for seed in seeds:
            if not isinstance(seed, dict) or "id" not in seed:
                raise ValueError("Seed vector needs an id")
            key = identifier(seed["id"])
            if key in vectors:
                raise ValueError("Duplicate seed id")
            vectors[key] = validate_schedule({k: v for k, v in seed.items() if k != "id"})
        if len({digest(v) for v in vectors.values()}) != len(vectors):
            raise ValueError("Duplicate seed vector")
        case_seeds = value.get("case_seeds", {case: list(vectors) for case in cases})
        if not isinstance(case_seeds, dict) or set(case_seeds) != set(cases):
            raise ValueError("Explicit case_seeds must cover every case")
        for case, ids in case_seeds.items():
            if (not isinstance(ids, list) or not ids or any(not isinstance(s, str) or s not in vectors for s in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Each case needs a nonempty unique subset of the declared seeds")
        if set().union(*(set(ids) for ids in case_seeds.values())) != set(vectors):
            raise ValueError("Declared seed vectors must be used by the matrix")
        expected = {(case, seed) for case in cases for seed in case_seeds[case]}
        self.schema_version = value["schema_version"]
        self.case_seeds = deepcopy(case_seeds)
        seen, normalized, inventory = set(), [], {str(self.path): sha(self.path)}
        physical_cases = {}
        for row in cells:
            if not isinstance(row, dict) or set(row) != {"case", "seed", "base_config", "baseline_run", "reference_run"}:
                raise ValueError("Cell requires case, seed, base_config, baseline_run and reference_run")
            key = (identifier(row["case"]), identifier(row["seed"]))
            if key in seen or key not in expected:
                raise ValueError("Duplicate or unknown matrix cell")
            seen.add(key)
            cell = deepcopy(row)
            for name in ("base_config", "baseline_run", "reference_run"):
                if name == "reference_run" and row[name] is None and not references_required:
                    continue
                if not isinstance(row[name], str) or not row[name]:
                    raise ValueError("Every scored cell needs its own qualified reference and input")
                p = Path(row[name]).expanduser()
                p = (self.path.parent / p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True)
                if (name == "base_config" and not p.is_file()) or (name != "base_config" and not p.is_dir()):
                    raise ValueError("Invalid matrix path: " + name)
                cell[name] = str(p)
            config = yaml.safe_load(Path(cell["base_config"]).read_text())
            if not isinstance(config, dict):
                raise ValueError("Cell base_config must be a mapping")
            identity = tuple(config.get(k) for k in ("design", "type", "pdk"))
            if not all(isinstance(v, str) and v for v in identity):
                raise ValueError("Cell needs a design, integration and PDK")
            if row["case"] in physical_cases and physical_cases[row["case"]] != identity:
                raise ValueError("Seeds of one case must use the same design/platform")
            physical_cases[row["case"]] = identity
            cell["seeds"] = vectors[row["seed"]]
            for stage, seed_name in (("pins", "io"), ("partition", "partition"),
                                     ("placement", "placement"), ("routing", "routing")):
                if config.get(stage, {}).get("seed") != cell["seeds"][seed_name]:
                    raise ValueError("Cell config seed differs: " + stage)
            paths = [Path(cell["base_config"])]
            from .evaluation import UPSTREAM_STAGE_DIRS
            paths += [Path(cell["baseline_run"]) / UPSTREAM_STAGE_DIRS[s] / "stage_manifest.json"
                      for s in spec.upstream_stages]
            if cell["reference_run"]:
                receipt_path = Path(cell["reference_run"]).parent / "evaluation_result.json"
                receipt = read(receipt_path)
                _require_reference_activity(spec, receipt.get("metrics", {}))
                contract = receipt.get("artifacts", {}).get("matrix_cell_contract", {})
                if (receipt.get("metrics", {}).get("runs_successfully") != 1
                        or receipt.get("artifacts", {}).get("matrix_reference") is not True
                        or contract.get("task") != spec.task_id or contract.get("seeds") != cell["seeds"]
                        or contract.get("base_config_sha256") != sha(cell["base_config"])
                        or contract.get("upstream") != {s: sha(Path(cell["baseline_run"]) /
                            UPSTREAM_STAGE_DIRS[s] / "stage_manifest.json") for s in spec.upstream_stages}):
                    raise ValueError("Unqualified or mismatched matrix reference: " + str(receipt_path))
                paths.append(receipt_path)
                if spec.kind in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
                    reference = Path(cell["reference_run"])
                    summary = reference / "reference_summary.json"
                    if summary.is_file():
                        from .packed_gnn_reference import validate_files
                        _, packed_paths = validate_files(reference, receipt)
                        paths.extend(packed_paths)
            inventory.update({str(p): sha(p) for p in paths})
            normalized.append(cell)
        if seen != expected:
            raise ValueError("Matrix must contain every case/seed pair exactly once")
        if len(set(physical_cases.values())) != len(physical_cases):
            raise ValueError("Duplicate physical case under different names")
        order = {key: i for i, key in enumerate((c, s) for c in cases for s in vectors)}
        self.cells = sorted(normalized, key=lambda c: order[(c["case"], c["seed"])])
        self.cases, self.seeds = list(cases), deepcopy(seeds)
        if spec.task_id in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}:
            from contest2 import scoring, storage
            for path in (scoring.REFERENCE_PATH, Path(scoring.__file__), Path(storage.__file__)):
                inventory[str(path)] = sha(path)
        self.inventory = inventory
        identity = {"schema": self.schema_version, "task": spec.task_id, "cells": self.cells,
                    "inventory": inventory, "aggregation": "equal-case-mean-of-paired-seed-scores"}
        if self.schema_version == 2:
            identity["case_seeds"] = self.case_seeds
        self.identity = digest(identity)
        self.spec = spec

    def assert_unchanged(self):
        if any(sha(path) != expected for path, expected in self.inventory.items()):
            raise ValueError("Frozen matrix/config/input/reference changed during search")

    def prompt(self):
        return ("\n\nEvaluation matrix: " + ", ".join(self.cases) + ".\nSeeds: " +
                "; ".join(s["id"] + "=" + str({k: v for k, v in s.items() if k != "id"}) for s in self.seeds) +
                ".\nRequired seeds by case: " + str(self.case_seeds) +
                ".\nEach candidate is evaluated on all " + str(len(self.cells)) + " case/seed cells. "
                "Each cell is verified against its qualified native initial-program reference. "
                "Maximize the mean of seed scores within each case, then the equal-weight mean across cases. " +
                ("Contest scoring uses common per-case WNS/TNS denominators from "
                 "contest2/manifests/common-timing-reference.json, after native verification. "
                 "Legal cell scores are max(raw_score, -100). Confirmed terminal physical failures "
                 "score -100 and remain in the fixed matrix mean, including final validation. "
                 "Missing or uncertain execution evidence is not a physical-failure score. "
                 "Raw native paired-reference scores are diagnostic only. " if self.spec.task_id in {
                    "tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"} else
                 "The unchanged reference scores 1. ") + "Inspect per-case results and seed variation; do not select "
                "a favorable seed. " + ("Any unconfirmed cell keeps the candidate unscored. "
                 if self.spec.task_id in {"tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}
                 else "Any failed cell makes the whole candidate unscored. ") +
                "One evaluation budget unit is one complete matrix, not one cell. "
                "The matrix replaces the single example case/seed settings in the task description.\n")

    def evaluate(self, candidate_dir, source, output, *, transport=None, workers=1,
                 timeout_seconds=21600, compact=False, qualify=False):
        from .evaluation import evaluate_task
        # A transport may build once before dispatch and return an immutable,
        # thread-safe cell callable. Never put image selection in process env.
        if type(workers) is not int or workers < 1:
            raise ValueError("matrix_workers must be positive")
        if workers > 1 and transport is None:
            raise ValueError("Parallel matrices require an isolated resource-bounded transport")
        self.assert_unchanged()
        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        write(root / "matrix.json", {"matrix_sha256": self.identity, "cases": self.cases,
              "seeds": self.seeds, "cells": self.cells, "qualification": qualify,
              "planned_native_cells": len(self.cells), "workers": workers})
        execute = transport or evaluate_task
        if hasattr(execute, "prepare_candidate"):
            execute = execute.prepare_candidate(self.spec.root, candidate_dir, source, root / "build")
        import threading
        uncertain = threading.Event()
        contest_scoring = not qualify and self.spec.task_id in {
            "tp_gnn_binary_loss_timing", "tp_gnn_loss_decoder_timing"}
        def one(cell):
            from grt_bench.runtime import UncertainExecution
            if uncertain.is_set():
                raise UncertainExecution("A preceding cell requires inspection")
            folder = root / "cells" / cell["case"] / cell["seed"]
            folder.mkdir(parents=True)
            try:
                self.assert_unchanged()
                result = execute(self.spec.root, candidate_dir, source, folder / "result",
                    baseline_run=Path(cell["baseline_run"]),
                    reference_run=None if qualify else Path(cell["reference_run"]),
                    base_config=Path(cell["base_config"]), seed_schedule=dict(cell["seeds"]),
                    create_routing_reference=qualify and self.spec.kind in ROUTING_SOURCE_TASK_KINDS,
                    keep_workspace=False, keep_runtime=False, compact_artifacts=compact and not qualify,
                    timeout_seconds=timeout_seconds)
                if result.metrics.get("runs_successfully") != 1:
                    raise ValueError(result.artifacts.get("error_message", "Native cell failed"))
                metrics = result.metrics
                if contest_scoring:
                    from contest2.scoring import score_metrics
                    metrics = score_metrics(cell["case"], metrics)
                score = finite(metrics.get("combined_score"), "local cell combined_score")
                if qualify:
                    _require_reference_activity(self.spec, result.metrics)
                self.assert_unchanged()
                row = {"case": cell["case"], "seed": cell["seed"], "status": "succeeded",
                       "score": None if qualify else score, "metrics": metrics,
                       "artifacts": result.artifacts, "result_dir": str(folder / "result")}
                if contest_scoring:
                    row.update(valid=True, raw_score=metrics["raw_score"])
            except Exception as error:
                from grt_bench.runtime import UncertainExecution
                if isinstance(error, UncertainExecution):
                    uncertain.set()
                    write(folder / "cell.json", {"case": cell["case"], "seed": cell["seed"],
                        "status": "uncertain", "error_type": type(error).__name__, "error_message": str(error)})
                    raise
                # Ordinary failures retain their detailed diagnostics.
                from .failure_feedback import native_failure_diagnostics
                row = {"case": cell["case"], "seed": cell["seed"], "status": "failed", "score": None,
                       "error_type": type(error).__name__, "error_message": str(error),
                       "result_dir": str(folder / "result")}
                if contest_scoring:
                    from contest2.scoring import PhysicalCellFailure, MIN_SCORE
                    if isinstance(error, PhysicalCellFailure):
                        # Integrity changes are never converted to a physical penalty.
                        self.assert_unchanged()
                        metrics = {**error.result.metrics, "combined_score": MIN_SCORE,
                                   "raw_score": None, "valid": False,
                                   "native_paired_combined_score": error.result.metrics.get("combined_score")}
                        row.update(status="penalized", score=MIN_SCORE, raw_score=None,
                                   valid=False, metrics=metrics, artifacts=error.result.artifacts,
                                   physical_failure_evidence=error.evidence)
                diagnostics = native_failure_diagnostics(folder / "result")
                if diagnostics:
                    row["native_failure_diagnostics"] = diagnostics
            write(folder / "cell.json", row)
            return row
        with ThreadPoolExecutor(max_workers=min(workers, len(self.cells))) as pool:
            rows = list(pool.map(one, self.cells))
        all_valid = all(row["status"] == "succeeded" for row in rows)
        complete = all(row["status"] in ({"succeeded", "penalized"} if contest_scoring else {"succeeded"})
                       for row in rows)
        case_results = []
        for case in self.cases:
            selected = [r for r in rows if r["case"] == case]
            scores = [r["score"] for r in selected if r["score"] is not None]
            stats = {} if len(scores) != len(selected) else {
                "mean": statistics.mean(scores), "stddev": statistics.pstdev(scores),
                "min": min(scores), "max": max(scores)}
            case_results.append({"case": case, "complete": all(r["score"] is not None for r in selected)
                                 if contest_scoring else all(r["status"] == "succeeded" for r in selected),
                                 "scores": {r["seed"]: r["score"] for r in selected}, **stats})
        summary = {"matrix_sha256": self.identity, "expected_cells": len(rows),
                   "succeeded_cells": sum(r["status"] == "succeeded" for r in rows),
                   "complete": complete, "qualification": qualify, "cases": case_results,
                   "cells": rows}
        score = statistics.mean(c["mean"] for c in case_results) if complete and not qualify else None
        metrics = {"runs_successfully": float(all_valid), "matrix": summary}
        if contest_scoring:
            summary.update(all_valid=all_valid, scored_cells=sum(r["score"] is not None for r in rows),
                           penalized_cells=sum(r["status"] == "penalized" for r in rows))
            metrics.update(evaluation_complete=float(complete), valid=all_valid)
        if score is not None:
            metrics["combined_score"] = score
        artifacts = {"matrix_result": str(root / "evaluation_result.json")}
        if not complete:
            artifacts.update(error_type="IncompleteMatrix", error_message="At least one case/seed cell failed; no aggregate score")
        result = EvaluationResult(metrics=metrics, artifacts=artifacts)
        write(root / "evaluation_result.json", result.to_dict())
        if qualify and complete:
            published = {"schema_version": self.schema_version, "task": self.spec.task_id, "cases": self.cases,
                         "seeds": self.seeds, "cells": []}
            if self.schema_version == 2:
                published["case_seeds"] = self.case_seeds
            for cell, row in zip(self.cells, rows):
                published["cells"].append({k: cell[k] for k in
                    ("case", "seed", "base_config", "baseline_run")})
                published["cells"][-1]["reference_run"] = str(Path(row["result_dir"]) / "run")
            write(root / "qualified_matrix.json", published)
        return result


def qualify_matrix(path, task_dir, source, output, *, transport=None, workers=1, timeout_seconds=21600):
    """Run each exact initial program once; qualification is separate from search."""
    from .program_bridge import initial_program_text, materialize_program
    from .model import load_candidate
    from .verifier import verify_candidate
    spec = load_task(task_dir)
    matrix = LocalEvaluationMatrix(path, spec, references_required=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    initial = materialize_program(spec, initial_program_text(spec, load_candidate(spec.initial_candidate)), source)
    verify_candidate(spec, initial, source)
    candidate = output / "initial"
    write(candidate / "candidate.json", initial)
    return matrix.evaluate(candidate, source, output / "qualification", transport=transport,
                           workers=workers, timeout_seconds=timeout_seconds, qualify=True)
