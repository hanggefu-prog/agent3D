from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from .benchmark_input import validate_benchmark_input
from .evostage_adapter import (
    EvolutionOptions,
    SUPPORTED_METHODS,
    _load_external,
    evostage_fingerprints,
    run_evolution,
)
from .llm_client import CodexExecLLM, OpenAICompatibleLLM, ReplayLLM, load_llm_config
from .model import (
    ROUTING_SOURCE_TASK_KINDS,
    load_candidate,
    load_task,
    resolve_task_path,
)
from .program_bridge import initial_program_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = PROJECT_ROOT / "local_tasks"
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
DEFAULT_LLM_CONFIGS = {
    "qwen": CONFIG_ROOT / "qwen.yaml",
    "openai": CONFIG_ROOT / "openai.yaml",
    "deepseek": CONFIG_ROOT / "deepseek.yaml",
}


def _llm_config_path(provider: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    return DEFAULT_LLM_CONFIGS[provider]


def _task_path(value: str) -> Path:
    return resolve_task_path(TASK_ROOT, value)


def _default_output(task: str, method: str) -> Path:
    root = Path(
        os.environ.get(
            "OPENAGENTIC3D_EVOLUTION_ROOT",
            str(Path.home() / ".local" / "state" / "openagentic3dbench" / "evolution"),
        )
    )
    return root.expanduser() / task / method / time.strftime("%Y%m%d-%H%M%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openagentic3dbench-evolve")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check an external EvoStage checkout")
    doctor.add_argument("--evostage-root", type=Path, required=True)
    llm = commands.add_parser("llm-check", help="Check LLM configuration without revealing the key")
    llm.add_argument(
        "--provider", choices=("qwen", "openai", "deepseek", "codex"), default="qwen"
    )
    llm.add_argument("--llm-config", type=Path)
    llm.add_argument("--live", action="store_true", help="Make one minimal API request")
    llm.add_argument("--codex-workspace", type=Path, default=PROJECT_ROOT)
    llm.add_argument("--codex-bin", default="codex")
    llm.add_argument("--codex-model")
    llm.add_argument("--codex-timeout", type=int, default=900)
    run = commands.add_parser("run", help="Run one EvoStage baseline on one local task")
    run.add_argument("task")
    run.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    run.add_argument("--evostage-root", type=Path, required=True)
    run.add_argument(
        "--provider",
        choices=("qwen", "openai", "deepseek", "codex", "replay"),
        default="qwen",
    )
    run.add_argument("--llm-config", type=Path)
    run.add_argument("--codex-workspace", type=Path, default=PROJECT_ROOT)
    run.add_argument("--codex-bin", default="codex")
    run.add_argument("--codex-model")
    run.add_argument("--codex-timeout", type=int, default=900)
    run.add_argument("--evaluation-mode", choices=("verify", "interface", "full"), default="full")
    run.add_argument("--open3dbench-root", type=Path)
    run.add_argument("--baseline-run", type=Path)
    run.add_argument(
        "--benchmark-input",
        type=Path,
        help="Validated frozen input bundle; preferred over the legacy --baseline-run",
    )
    run.add_argument("--reference-run", type=Path)
    run.add_argument("--matrix", type=Path)
    run.add_argument("--matrix-workers", type=int, default=1)
    run.add_argument("--matrix-runtime", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--max-samples", type=int, default=50,
        help="Total generated-candidate/evaluation budget (default: 50)",
    )
    run.add_argument(
        "--max-generations", type=int, default=50,
        help="Safety ceiling; max-samples normally binds first (default: 50)",
    )
    run.add_argument("--population-size", type=int, default=5)
    run.add_argument(
        "--compact-evaluations",
        action="store_true",
        help=(
            "Retain candidate/config/metrics/logs but delete reproducible checkout, "
            "build, and copied run trees after every evaluation"
        ),
    )
    run.add_argument("--timeout", type=int, default=21600)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        root = args.evostage_root.expanduser().resolve()
        external = _load_external(root)
        fingerprints = evostage_fingerprints(root)
        print(json.dumps({
            "status": "ok",
            "root": str(root),
            "methods": list(SUPPORTED_METHODS),
            "loaded_classes": sorted(key for key in external if key != "Evaluation"),
            "fingerprints": fingerprints,
            "generated_code_execution": "disabled by task-specific AST bridge",
            "embedded_evostage_llm": "disabled and replaced by selected provider",
        }, indent=2, sort_keys=True))
        return 0

    if args.command == "llm-check":
        if args.provider == "codex":
            output = Path(os.environ.get("OPENAGENTIC3D_CODEX_CHECK_ROOT", "/tmp"))
            client = CodexExecLLM(
                args.codex_workspace,
                output / "openagentic3d-codex-check-{}".format(time.strftime("%Y%m%d-%H%M%S")),
                codex_binary=args.codex_bin,
                model=args.codex_model,
                timeout_seconds=args.codex_timeout,
            )
            report = client.public_dict()
            report["status"] = "configured"
            if args.live:
                response = client.draw_sample("Reply with exactly: OK")
                report["live_request"] = "passed" if response.strip() else "empty"
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        config = load_llm_config(_llm_config_path(args.provider, args.llm_config))
        if config.provider != args.provider:
            raise ValueError("LLM config provider does not match --provider")
        report = config.public_dict()
        report["status"] = "configured" if report["api_key_configured"] else "missing_api_key"
        if args.live:
            client = OpenAICompatibleLLM(config)
            response = client.draw_sample("Reply with exactly: OK")
            client.close()
            report["live_request"] = "passed" if response.strip() else "empty"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["api_key_configured"] or not args.live else 1

    spec = load_task(_task_path(args.task))
    if not args.evostage_root.expanduser().is_dir():
        raise FileNotFoundError(args.evostage_root)
    if args.evaluation_mode != "verify" or spec.program_mode == "task_adapter":
        if args.open3dbench_root is None or not args.open3dbench_root.expanduser().is_dir():
            raise ValueError(
                "--open3dbench-root must name an existing checkout for this task/mode"
            )
    if args.benchmark_input is not None and args.baseline_run is not None:
        raise ValueError("use --benchmark-input or legacy --baseline-run, not both")
    baseline_run = args.baseline_run
    benchmark_report = None
    base_config = None
    if args.benchmark_input is not None:
        benchmark_report = validate_benchmark_input(
            args.benchmark_input, task_id=spec.task_id
        )
        baseline_run = Path(benchmark_report["run_dir"])
        base_config = Path(benchmark_report["config"])
    if args.evaluation_mode == "full" and spec.upstream_stages and args.matrix is None:
        if baseline_run is None or not baseline_run.expanduser().is_dir():
            requirement = "--benchmark-input"
            if spec.benchmark_input_id:
                requirement += " {}".format(spec.benchmark_input_id)
            raise ValueError(
                "{} is required for this full task; --baseline-run is legacy-compatible".format(
                    requirement
                )
            )
    if args.reference_run is not None and not args.reference_run.expanduser().is_dir():
        raise ValueError("--reference-run must exist")
    if (
        args.evaluation_mode == "full"
        and spec.kind in ROUTING_SOURCE_TASK_KINDS
        and args.reference_run is None
        and args.matrix is None
    ):
        raise ValueError(
            "--reference-run is required for full routing source evolution"
        )
    transport = None
    if args.matrix is not None:
        if args.evaluation_mode != "full" or any((args.baseline_run, args.reference_run, args.benchmark_input)):
            raise ValueError("Matrix replaces single-cell paths and requires full mode")
        from .matrix import LocalEvaluationMatrix
        LocalEvaluationMatrix(args.matrix, spec)
        if args.matrix_runtime:
            from .matrix_docker import DockerMatrixTransport
            transport = DockerMatrixTransport(**json.loads(args.matrix_runtime.read_text()))
    elif args.matrix_runtime or args.matrix_workers != 1:
        raise ValueError("Matrix runtime/workers require --matrix")
    output = (args.output_dir or _default_output(spec.task_id, args.method)).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "history.jsonl").exists():
        raise RuntimeError("output directory already contains an evolution run: {}".format(output))
    initial = load_candidate(spec.initial_candidate)
    if args.provider in {"qwen", "openai", "deepseek"}:
        config = load_llm_config(_llm_config_path(args.provider, args.llm_config))
        if config.provider != args.provider:
            raise ValueError("LLM config provider does not match --provider")
        provider = OpenAICompatibleLLM(config)
        provider_public = config.public_dict()
    elif args.provider == "codex":
        provider = CodexExecLLM(
            args.codex_workspace,
            output / "codex_samples",
            codex_binary=args.codex_bin,
            model=args.codex_model,
            timeout_seconds=args.codex_timeout,
        )
        provider_public = provider.public_dict()
    else:
        provider = ReplayLLM(
            initial_program_text(spec, initial)
            if spec.program_mode == "task_adapter"
            else initial
        )
        provider_public = {"provider": "replay", "network": False, "api_key_configured": False}
    options = EvolutionOptions(
        task_dir=spec.root,
        evostage_root=args.evostage_root,
        output_dir=output,
        evaluation_mode=args.evaluation_mode,
        open3dbench_root=args.open3dbench_root,
        baseline_run=baseline_run,
        reference_run=args.reference_run,
        base_config=base_config,
        benchmark_id=benchmark_report["benchmark_id"] if benchmark_report else None,
        evaluation_matrix=args.matrix, matrix_workers=args.matrix_workers,
        compact_evaluations=args.compact_evaluations,
        timeout_seconds=args.timeout,
    )
    run_config = {
        "schema_version": 1,
        "matrix": str(args.matrix.resolve()) if args.matrix else None,
        "matrix_workers": args.matrix_workers,
        "task": spec.task_id,
        "method": args.method,
        "provider": provider_public,
        "evostage": {
            "root": str(args.evostage_root.expanduser().resolve()),
            "fingerprints": evostage_fingerprints(args.evostage_root),
        },
        "evaluation_mode": args.evaluation_mode,
        "open3dbench_root": str(args.open3dbench_root.resolve()) if args.open3dbench_root else None,
        "benchmark_input": benchmark_report["manifest"] if benchmark_report else None,
        "benchmark_id": benchmark_report["benchmark_id"] if benchmark_report else None,
        "baseline_run": str(baseline_run.resolve()) if baseline_run else None,
        "reference_run": str(args.reference_run.resolve()) if args.reference_run else None,
        "base_config": str(base_config.resolve()) if base_config else str(spec.base_config),
        "max_samples": args.max_samples,
        "max_generations": args.max_generations,
        "population_size": args.population_size,
        "compact_evaluations": args.compact_evaluations,
        "timeout_seconds": args.timeout,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        summary = run_evolution(
            args.method,
            provider,
            options,
            max_samples=args.max_samples,
            max_generations=args.max_generations,
            pop_size=args.population_size,
            **({"evaluation_transport": transport} if transport is not None else {}),
        )
    finally:
        provider.close()
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
