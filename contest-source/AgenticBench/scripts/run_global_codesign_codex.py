#!/usr/bin/env python3
"""Run the reproducible Codex candidate-program baseline for global co-design tasks.

This is a benchmark adapter, not the global-task API: the task itself remains a
scope-free workspace problem.  For matched first results, this adapter presents
the maintained TP-GNN/concurrent-DREAMPlace intervention surface to one
persistent Codex thread and spends only trusted final evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_bench.model import load_global_task
from local_bench.benchmark_input import validate_benchmark_input
from local_bench.evolve_protocol import render_candidate_program
from local_bench.model import load_task
from local_bench.prompting import build_problem_description
from local_bench.verifier import verify_candidate
from scripts.run_sta_codesign_comparison import (
    PROGRAM_SCHEMA,
    PersistentCodex,
    _check_source_patch_applies,
    _program_candidate,
)
import scripts.run_sta_codesign_comparison as remote_evaluation


GLOBAL_TASK_ROOT = ROOT / "global_tasks"
PROFILE_ROOT = ROOT / "evaluation_profiles"
TASKS = {
    "lol_partition_placement_cts_thermal": {
        "profile": "tp_gnn_placement_cts_thermal",
        "target": "tp_gnn_placement_cts_thermal",
        "label": "two-pass OPT + CTS cell-power HotSpot average temperature",
    },
    "lol_partition_placement_timing": {
        "profile": "tp_gnn_placement_two_pass",
        "target": "tp_gnn_placement_sta",
        "label": "two-pass setup STA",
    },
    "lol_partition_placement_thermal": {
        "profile": "tp_gnn_placement_thermal",
        "target": "tp_gnn_placement_thermal",
        "label": "full-flow HotSpot average temperature",
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _result(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation result is not an object: {}".format(path))
    return value


def _score(value: Mapping[str, Any]) -> Optional[float]:
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("runs_successfully") != 1.0:
        return None
    score = metrics.get("combined_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return float(score) if math.isfinite(float(score)) else None


def _compact(value: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = value.get("metrics")
    artifacts = value.get("artifacts")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    keep = (
        "combined_score",
        "normalized_sta_cost",
        "normalized_wns_cost",
        "normalized_tns_cost",
        "final_global_wns_setup_ns",
        "final_global_tns_setup_ns",
        "global_average_temperature_k",
        "global_average_temperature_c",
        "bottom_average_temperature_k",
        "top_average_temperature_k",
        "global_maximum_temperature_c",
        "normalized_average_temperature_cost",
        "total_power_w",
        "cut_net_count",
        "area_balance_error",
        "placement_total_hpwl_um",
        "placement_overflow_bottom",
        "placement_overflow_top",
        "runs_successfully",
    )
    return {
        "metrics": {key: metrics[key] for key in keep if key in metrics},
        "error_type": artifacts.get("error_type"),
        "error_message": artifacts.get("error_message"),
    }


def _evaluate(
    args: argparse.Namespace,
    *,
    target: str,
    candidate: Path,
    output: Path,
    reference: Optional[Union[Path, str]],
) -> Tuple[Dict[str, Any], Union[Path, str]]:
    if args.evaluation_host:
        relative = output.relative_to(args.output_dir).as_posix()
        remote_root = args.remote_output_root.rstrip("/") + "/" + relative
        result, run = remote_evaluation.evaluate_remote(
            task=target,
            local_candidate=candidate,
            remote_root=remote_root,
            input_kind="bundle",
            input_path=args.remote_benchmark_input,
            reference_run=str(reference) if reference is not None else "-",
            gpu=args.gpu,
            retention="full",
            log_path=output.parent / (output.name + ".log"),
        )
        return result, run
    if output.exists():
        result_path = output / "evaluation_result.json"
        if not result_path.is_file():
            raise ValueError("incomplete evaluation directory requires inspection: {}".format(output))
        return _result(result_path), output / "run"
    command = [
        str(args.helper),
        target,
        str(candidate),
        str(output),
        "bundle",
        str(args.benchmark_input),
        str(reference) if reference is not None else "-",
        str(args.gpu),
        "full",
        str(args.profile_config),
    ]
    log_path = output.parent / (output.name + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    environment = os.environ.copy()
    process = subprocess.run(
        command,
        cwd=str(args.source_root),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=args.evaluation_timeout,
    )
    log_path.write_text(process.stdout or "", encoding="utf-8")
    result_path = output / "evaluation_result.json"
    if not result_path.is_file():
        raise RuntimeError(
            "evaluation exited {} without a result; see {}".format(process.returncode, log_path)
        )
    result = _result(result_path)
    result["baseline_driver"] = {
        "exit_code": process.returncode,
        "wall_seconds": time.time() - started,
        "log": str(log_path),
    }
    _write_json(output / "evaluation_result.driver.json", result)
    return result, output / "run"


def _initial_candidate(spec, destination: Path) -> Path:
    candidate = json.loads((spec.initial_candidate / "candidate.json").read_text(encoding="utf-8"))
    verify_candidate(spec, candidate)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "candidate.json", candidate)
    (destination / "program.py").write_text(render_candidate_program(candidate), encoding="utf-8")
    return destination


def run(args: argparse.Namespace) -> None:
    definition = TASKS[args.task_id]
    global_task = load_global_task(GLOBAL_TASK_ROOT / args.task_id)
    if args.evaluation_host:
        remote_evaluation.REMOTE_HOST = args.evaluation_host
        remote_evaluation.REMOTE_BUNDLE = args.remote_benchmark_input
        remote_evaluation.REMOTE_COMPARISON = args.remote_output_root
        remote_evaluation.REMOTE_HELPER = args.remote_helper
        remote_evaluation.REMOTE_PROFILE_CONFIG = args.remote_profile_config
        remote_evaluation.SSH_PROXY_COMMAND = args.ssh_proxy_command
        remote_evaluation.REMOTE_ENV = tuple(args.remote_env)
        bundle_manifest = remote_evaluation._remote_result(
            args.remote_benchmark_input.rstrip("/") + "/benchmark_input.json"
        )
        benchmark_id = str(bundle_manifest.get("benchmark_id", ""))
        if not benchmark_id:
            raise ValueError("remote benchmark input has no benchmark_id")
        bundle = {"benchmark_id": benchmark_id}
    else:
        assert args.benchmark_input is not None
        bundle = validate_benchmark_input(
            args.benchmark_input, task_id=definition["target"]
        )
    profile = replace(
        load_task(PROFILE_ROOT / definition["profile"]),
        base_config=args.profile_config,
        benchmark_input_id=str(bundle["benchmark_id"]),
    )
    root = args.output_dir.expanduser().resolve()
    if args.resume:
        config = _result(root / "run_config.json")
        if config.get("task_id") != args.task_id or int(config.get("rounds", -1)) != args.rounds:
            raise ValueError("resume configuration differs from the saved baseline")
    else:
        root.mkdir(parents=True, exist_ok=False)
        _write_json(
            root / "run_config.json",
            {
                "schema_version": 1,
                "task_id": args.task_id,
                "task_digest": global_task.digest(),
                "adapter": "persistent-codex-candidate-program-v1",
                "adapter_scope": list(profile.allowed_source_paths),
                "model": args.model,
                "reasoning_effort": "xhigh",
                "rounds": args.rounds,
                "gpu": args.gpu,
                "benchmark_input": (
                    str(args.benchmark_input)
                    if args.benchmark_input is not None
                    else None
                ),
                "benchmark_id": bundle["benchmark_id"],
                "source_root": str(args.source_root),
                "profile_config": str(args.profile_config),
                "evaluation_host": args.evaluation_host,
                "remote_benchmark_input": args.remote_benchmark_input,
                "remote_reference_run": args.remote_reference_run,
                "remote_output_root": args.remote_output_root,
                "remote_helper": args.remote_helper,
                "remote_profile_config": args.remote_profile_config,
                "remote_env": list(args.remote_env),
                "objective": global_task.objective.kind,
                "expensive_budget_semantics": "one trusted final evaluation per round; no proxies",
            },
        )

    if args.evaluation_host:
        reference: Optional[Union[Path, str]] = args.remote_reference_run
    else:
        reference = args.reference_run
    if reference is None:
        reference_candidate = _initial_candidate(profile, root / "reference/candidate")
        reference_result, reference_run = _evaluate(
            args,
            target=definition["target"],
            candidate=reference_candidate,
            output=root / "reference/evaluation",
            reference=None,
        )
        if _score(reference_result) != 1.0:
            raise ValueError("fresh reference failed self-normalized trusted evaluation")
        reference = reference_run
    if not args.evaluation_host and not (Path(reference) / "run.json").is_file():
        raise FileNotFoundError("reference run: {}".format(reference))

    prompt = build_problem_description(
        profile,
        json.loads((profile.initial_candidate / "candidate.json").read_text(encoding="utf-8")),
        args.source_root,
    )
    agent = PersistentCodex(
        name="codex-global-codesign",
        root=root / "agent",
        schema=PROGRAM_SCHEMA,
        model=args.model,
        codex_binary=str(args.codex_bin),
        workspace=args.source_root,
        timeout_seconds=args.codex_timeout,
    )
    history_path = root / "history.json"
    history = _result(history_path).get("history", []) if history_path.is_file() else []
    if not isinstance(history, list) or len(history) > args.rounds:
        raise ValueError("saved history is malformed")
    incumbent_score = 1.0
    incumbent_program = render_candidate_program(
        json.loads((profile.initial_candidate / "candidate.json").read_text(encoding="utf-8"))
    )
    for row in history:
        if row.get("accepted_as_incumbent"):
            incumbent_score = float(row["score"])
            incumbent_program = (root / "rounds/{:02d}/candidate/program.py".format(int(row["round"]))).read_text(encoding="utf-8")

    for number in range(len(history) + 1, args.rounds + 1):
        if number == 1:
            request = (
                "You are the persistent single-Codex baseline for the global co-design "
                "task below. First form an explicit plan inside the hypothesis, then "
                "return one complete candidate program. You receive no proxy score and "
                "exactly one trusted {} evaluation this round. Do not alter the fixed "
                "evaluator, reference, legality gates, or downstream model.\n\n"
                "GLOBAL TASK:\n{}\n\nCANDIDATE ADAPTER CONTRACT:\n{}"
            ).format(definition["label"], global_task.instruction, prompt)
        else:
            request = (
                "Continue the same global co-design task. Start with an explicit next-step "
                "plan in the hypothesis, then return one materially revised complete "
                "candidate program. Use only the trusted final feedback below; there are "
                "no hidden proxy evaluations.\n\n{}"
            ).format(
                json.dumps(
                    {
                        "history": history,
                        "incumbent_score": incumbent_score,
                        "incumbent_program": incumbent_program,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        round_root = root / "rounds/{:02d}".format(number)
        candidate_root = round_root / "candidate"
        candidate_path = candidate_root / "candidate.json"
        response_path = candidate_root / "agent_response.json"
        program_path = candidate_root / "program.py"
        if candidate_path.is_file() and response_path.is_file() and program_path.is_file():
            # A trusted evaluation can be retried after an infrastructure failure
            # without spending another model call or changing the candidate.
            _candidate = _result(candidate_path)
            verify_candidate(profile, _candidate)
            _check_source_patch_applies(
                args.source_root, str(_candidate.get("source_patch", "")), candidate_root
            )
            response = _result(response_path)
            program = program_path.read_text(encoding="utf-8")
        else:
            program, _candidate, response = _program_candidate(
                agent, profile, request, candidate_root
            )
        result, _candidate_run = _evaluate(
            args,
            target=definition["target"],
            candidate=round_root / "candidate",
            output=round_root / "evaluation",
            reference=reference,
        )
        score = _score(result)
        accepted = score is not None and score > incumbent_score
        if accepted:
            incumbent_score = float(score)
            incumbent_program = program
        row = {
            "round": number,
            "hypothesis": response.get("hypothesis"),
            "score": score,
            "accepted_as_incumbent": accepted,
            "reference_run": str(reference),
            **_compact(result),
        }
        history.append(row)
        _write_json(history_path, {"history": history})
        print(json.dumps({"event": "round_complete", **row}, ensure_ascii=False), flush=True)
    _write_json(
        root / "summary.json",
        {
            "schema_version": 1,
            "task_id": args.task_id,
            "model": args.model,
            "reference_run": str(reference),
            "final_evaluations": len(history),
            "incumbent_score": incumbent_score,
            "history": history,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", choices=tuple(TASKS))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--benchmark-input", type=Path)
    parser.add_argument("--profile-config", required=True, type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--helper", type=Path, default=ROOT / "skills/open3dbench-sta-evaluation/scripts/run_remote_sta.sh")
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--codex-bin", type=Path, default=Path.home() / ".local/bin/codex")
    parser.add_argument("--codex-timeout", type=int, default=3600)
    parser.add_argument("--evaluation-timeout", type=int, default=90000)
    parser.add_argument("--evaluation-host")
    parser.add_argument("--remote-benchmark-input")
    parser.add_argument("--remote-reference-run")
    parser.add_argument("--remote-output-root")
    parser.add_argument("--remote-helper")
    parser.add_argument("--remote-profile-config")
    parser.add_argument(
        "--remote-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Non-secret environment setting passed to the remote evaluation helper.",
    )
    parser.add_argument("--ssh-proxy-command")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    if args.benchmark_input is not None:
        args.benchmark_input = args.benchmark_input.expanduser().resolve()
    args.profile_config = args.profile_config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.helper = args.helper.expanduser().resolve()
    args.codex_bin = args.codex_bin.expanduser().resolve()
    if args.reference_run is not None:
        args.reference_run = args.reference_run.expanduser().resolve()
    if args.evaluation_host:
        required = {
            "remote_benchmark_input": args.remote_benchmark_input,
            "remote_output_root": args.remote_output_root,
            "remote_helper": args.remote_helper,
            "remote_profile_config": args.remote_profile_config,
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ValueError("remote evaluation requires {}".format(", ".join(missing)))
        if args.reference_run is not None:
            raise ValueError("use --remote-reference-run with --evaluation-host")
        if args.benchmark_input is not None:
            raise ValueError("use --remote-benchmark-input with --evaluation-host")
        safe_remote_env = re.compile(r"^[A-Z_][A-Z0-9_]*=[A-Za-z0-9_./:+-]+$")
        invalid_remote_env = [value for value in args.remote_env if not safe_remote_env.fullmatch(value)]
        if invalid_remote_env:
            raise ValueError("unsafe --remote-env value: {}".format(invalid_remote_env[0]))
    elif any(
        value
        for value in (
            args.remote_benchmark_input,
            args.remote_reference_run,
            args.remote_output_root,
            args.remote_helper,
            args.remote_profile_config,
            args.ssh_proxy_command,
            *args.remote_env,
        )
    ):
        raise ValueError("remote evaluation options require --evaluation-host")
    elif args.benchmark_input is None:
        raise ValueError("local evaluation requires --benchmark-input")
    if args.rounds < 1 or args.rounds > 5:
        raise ValueError("Codex baseline uses between one and the task's five final evaluations")
    if args.gpu not in range(8):
        raise ValueError("GPU must be between 0 and 7")
    directories = [args.source_root]
    if args.benchmark_input is not None:
        directories.append(args.benchmark_input)
    for path in directories:
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (args.profile_config, args.helper, args.codex_bin):
        if not path.is_file():
            raise FileNotFoundError(path)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
