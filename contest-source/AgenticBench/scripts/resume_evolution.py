#!/usr/bin/env python3
"""Resume an interrupted EvoStage run into a new, auditable output directory.

The normal evolution CLI intentionally refuses non-empty output directories.
This helper preserves that invariant: it reads an interrupted run, copies its
history and prompt trace into a fresh directory, restores feedback/cache state,
seeds previously scored programs back into the EvoStage population, and spends
only the remaining candidate budget.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for dependency in os.environ.get("OPENAGENTIC3D_EXTRA_PYTHONPATH", "").split(
    os.pathsep
):
    if dependency and dependency not in sys.path:
        # Append instead of prepending so the runtime image's binary NumPy wins
        # over pure-Python dependency bundles prepared for other interpreters.
        sys.path.append(dependency)

from local_bench import evostage_adapter
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.evolve_cli import _task_path
from local_bench.llm_client import OpenAICompatibleLLM, load_llm_config
from local_bench.prompting import ContextualLLM
from local_bench.search_api import ProgramEvaluationHarness


def _json_lines(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("{}:{} is not a JSON object".format(path, line_number))
        rows.append(value)
    return rows


def _write_json_lines(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _response_hashes(trace_rows: Iterable[Dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in trace_rows:
        attempts = row.get("provider_attempts", [])
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, dict) and attempt.get("response_sha256"):
                    hashes.add(str(attempt["response_sha256"]))
        response = row.get("response")
        if isinstance(response, str) and response:
            hashes.add(hashlib.sha256(response.encode("utf-8")).hexdigest())
    return hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue an interrupted EvoStage run without re-evaluating prior samples."
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-config", type=Path, required=True)
    parser.add_argument("--total-samples", type=int, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate resume provenance and paths without creating output or calling the LLM.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = args.resume_from.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    history_path = source / "history.jsonl"
    trace_path = source / "prompt_trace.jsonl"
    config_path = source / "run_config.json"
    for path in (history_path, trace_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output == source:
        raise ValueError("resume output must differ from the interrupted run")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("resume output directory is not empty: {}".format(output))

    previous_records = _json_lines(history_path)
    previous_trace = _json_lines(trace_path)
    if not previous_records:
        raise ValueError("interrupted run has no history to resume")
    if args.total_samples <= len(previous_records):
        raise ValueError(
            "total sample budget {} does not exceed completed records {}".format(
                args.total_samples, len(previous_records)
            )
        )
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict):
        raise ValueError("run_config.json must contain an object")
    if source_config.get("method") != "evostage":
        raise ValueError("resume helper currently supports only EvoStage runs")
    if source_config.get("evaluation_mode") != "full":
        raise ValueError("resume helper requires a full physical-evaluation run")

    benchmark_input = Path(str(source_config["benchmark_input"])).resolve()
    benchmark = validate_benchmark_input(
        benchmark_input, task_id=str(source_config["task"])
    )
    reference_run = Path(str(source_config["reference_run"])).resolve()
    open3dbench_root = Path(str(source_config["open3dbench_root"])).resolve()
    evostage_root = Path(str(source_config["evostage"]["root"])).resolve()
    for path in (reference_run, open3dbench_root, evostage_root):
        if not path.is_dir():
            raise FileNotFoundError(path)

    remaining = args.total_samples - len(previous_records)
    response_hashes = _response_hashes(previous_trace)
    max_trace_order = max(
        (int(row.get("order", 0)) for row in previous_trace), default=0
    )
    trace_by_order = {
        int(row["order"]): row
        for row in previous_trace
        if isinstance(row.get("order"), int)
    }
    missing_trace = sorted(
        int(row.get("order", 0))
        for row in previous_records
        if isinstance(row.get("score"), (int, float))
        and int(row.get("order", 0)) not in trace_by_order
    )
    if missing_trace:
        raise ValueError(
            "scored history records have no prompt trace: {}".format(missing_trace)
        )
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "source_run": str(source),
                    "output_dir": str(output),
                    "completed_records": len(previous_records),
                    "completed_scored_records": sum(
                        isinstance(row.get("score"), (int, float))
                        for row in previous_records
                    ),
                    "prior_prompt_records": len(previous_trace),
                    "remaining_samples": remaining,
                    "target_total_samples": args.total_samples,
                    "benchmark_id": benchmark["benchmark_id"],
                    "reference_run": str(reference_run),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)

    class ResumedHarness(ProgramEvaluationHarness):
        def __init__(self, options):
            super().__init__(options)
            with self._lock:
                self._records = copy.deepcopy(previous_records)
                self._cache = {
                    str(row["candidate_sha256"]): copy.deepcopy(row)
                    for row in previous_records
                    if row.get("candidate_sha256") and row.get("score") is not None
                }
            _write_json_lines(self.output_dir / "history.jsonl", previous_records)

    class ResumedContextualLLM(ContextualLLM):
        def __init__(self, provider, harness, destination, problem_description=""):
            super().__init__(provider, harness, destination, problem_description)
            self._counter = max_trace_order
            self._response_hashes.update(response_hashes)
            _write_json_lines(destination, previous_trace)

    original_harness = evostage_adapter.EvolutionHarness
    original_context = evostage_adapter.ContextualLLM
    original_seed = evostage_adapter._seed_frozen_reference
    restore_info = {"population_candidates": 0}

    def seed_reference_and_resumed_candidates(method, harness):
        seeded = original_seed(method, harness)
        functions = getattr(method, "_function_to_evolve_list", ())
        populations = getattr(method, "_population_list", ())
        if len(functions) != 1 or len(populations) != 1:
            raise RuntimeError("resume helper requires one evolved function and population")
        prior_scored = sorted(
            (
                row
                for row in previous_records
                if isinstance(row.get("score"), (int, float))
                and int(row.get("order", 0)) in trace_by_order
            ),
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        population_limit = max(0, int(source_config.get("population_size", 5)) - 1)
        restored = 0
        for row in prior_scored[:population_limit]:
            trace = trace_by_order[int(row["order"])]
            body = trace.get("transport_response")
            if not isinstance(body, str) or not body.strip():
                continue
            function = copy.deepcopy(functions[0])
            function.body = body.strip("\n")
            function.score = float(row["score"])
            function.algorithm = "Resumed evaluated candidate order {}".format(row["order"])
            function.evaluate_time = float(row.get("runtime_seconds", 0.0))
            function.sample_time = 0.0
            populations[0].register_function(function)
            restored += 1
        restore_info["population_candidates"] = restored
        return seeded

    evostage_adapter.EvolutionHarness = ResumedHarness
    evostage_adapter.ContextualLLM = ResumedContextualLLM
    evostage_adapter._seed_frozen_reference = seed_reference_and_resumed_candidates

    llm_config = load_llm_config(args.llm_config.expanduser().resolve())
    provider = OpenAICompatibleLLM(llm_config)
    options = evostage_adapter.EvolutionOptions(
        task_dir=_task_path(str(source_config["task"])),
        evostage_root=evostage_root,
        output_dir=output,
        evaluation_mode="full",
        open3dbench_root=open3dbench_root,
        baseline_run=Path(str(benchmark["run_dir"])).resolve(),
        reference_run=reference_run,
        base_config=Path(str(benchmark["config"])).resolve(),
        benchmark_id=str(benchmark["benchmark_id"]),
        compact_evaluations=bool(source_config.get("compact_evaluations", False)),
        timeout_seconds=int(source_config.get("timeout_seconds", 21600)),
    )
    resumed_config = copy.deepcopy(source_config)
    resumed_config.update(
        {
            "resume": {
                "source_run": str(source),
                "completed_records": len(previous_records),
                "remaining_samples": remaining,
                "target_total_samples": args.total_samples,
                "restores_feedback": True,
                "restores_duplicate_detection": True,
                "restores_evaluation_cache": True,
                "restores_scored_population": True,
            },
            "output_dir": str(output),
            "max_samples": args.total_samples,
        }
    )
    (output / "run_config.json").write_text(
        json.dumps(resumed_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
        summary = evostage_adapter.run_evolution(
            "evostage",
            provider,
            options,
            max_samples=remaining,
            max_generations=int(source_config.get("max_generations", args.total_samples)),
            pop_size=int(source_config.get("population_size", 5)),
        )
        summary["max_samples"] = args.total_samples
        summary["resume"] = resumed_config["resume"]
        summary["resume"]["population_candidates_restored"] = restore_info[
            "population_candidates"
        ]
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if summary.get("accepted") else 1
    finally:
        provider.close()
        evostage_adapter.EvolutionHarness = original_harness
        evostage_adapter.ContextualLLM = original_context
        evostage_adapter._seed_frozen_reference = original_seed


if __name__ == "__main__":
    raise SystemExit(main())
