from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .benchmark_input import (
    load_benchmark_input_profile,
    prepare_benchmark_input,
    resolve_benchmark_run,
    validate_benchmark_input,
)
from .evaluation import evaluate_task
from .model import load_candidate, load_task, resolve_task_path
from .program_bridge import initial_program_text, materialize_program
from .prompting import export_openevolve_bundle, task_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ROOT = PROJECT_ROOT / "local_tasks"


def _task_path(value: str) -> Path:
    return resolve_task_path(DEFAULT_TASK_ROOT, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m local_bench")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List local optimization tasks and their readiness")
    listing.add_argument("--core", action="store_true", help="List the eight current research tasks")
    inspect = commands.add_parser("inspect", help="Show one task contract")
    inspect.add_argument("task")
    evaluate = commands.add_parser("evaluate", help="Evaluate one candidate")
    evaluate.add_argument("task")
    evaluate.add_argument("--candidate", type=Path)
    evaluate.add_argument("--open3dbench-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--baseline-run", type=Path)
    evaluate.add_argument(
        "--benchmark-input",
        type=Path,
        help="Validated frozen input bundle; preferred over the legacy --baseline-run",
    )
    evaluate.add_argument("--reference-run", type=Path)
    evaluate.add_argument(
        "--base-config",
        type=Path,
        help="Explicit case configuration paired with the frozen benchmark input",
    )
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--discard-workspace", action="store_true")
    evaluate.add_argument("--discard-runtime", action="store_true")
    evaluate.add_argument(
        "--compact-artifacts",
        action="store_true",
        help="Delete reproducible workspace, build, and run trees after scoring",
    )
    evaluate.add_argument(
        "--create-routing-reference",
        action="store_true",
        help=(
            "Bootstrap an unscaled maintained-policy GRT run whose raw metrics "
            "can be used as --reference-run"
        ),
    )
    evaluate.add_argument("--timeout", type=int, default=21600)
    evaluate.add_argument("--matrix", type=Path, help="Qualified case/seed matrix; replaces single-cell paths")
    evaluate.add_argument("--matrix-workers", type=int, default=1)
    evaluate.add_argument("--matrix-runtime", type=Path, help="JSON Docker transport settings for the shared resource queue")
    qualify = commands.add_parser("qualify-matrix", help="Qualify the initial program on every local matrix cell")
    qualify.add_argument("task")
    qualify.add_argument("--matrix", type=Path, required=True)
    qualify.add_argument("--open3dbench-root", type=Path, required=True)
    qualify.add_argument("--output-dir", type=Path, required=True)
    qualify.add_argument("--matrix-workers", type=int, default=1)
    qualify.add_argument("--matrix-runtime", type=Path)
    qualify.add_argument("--timeout", type=int, default=21600)
    export = commands.add_parser(
        "export-openevolve",
        help="Create a native OpenEvolve bundle with resolved task and source context",
    )
    export.add_argument("task")
    export.add_argument("--open3dbench-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument(
        "--benchmark-profile",
        help="Bind the exported prompt contract to one benchmark-input profile",
    )
    prepare = commands.add_parser(
        "prepare-input",
        help="Generate one versioned input/floorplan/partition bundle",
    )
    prepare.add_argument("profile")
    prepare.add_argument("--open3dbench-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--python", type=Path)
    prepare.add_argument("--dreamplace-install", type=Path)
    prepare.add_argument(
        "--seed-run",
        type=Path,
        help="Reuse validated input/floorplan stages and regenerate partition",
    )
    prepare.add_argument("--timeout", type=int, default=21600)
    prepare.add_argument("--resume", action="store_true")
    validate = commands.add_parser(
        "validate-input", help="Verify every frozen stage artifact and its hash"
    )
    validate.add_argument("input", type=Path)
    validate.add_argument("--task")
    render = commands.add_parser(
        "render-matrix",
        help="Render an unqualified case/seed matrix from a declarative matrix profile",
    )
    render.add_argument("task")
    render.add_argument(
        "--baseline-runs",
        type=Path,
        required=True,
        help="JSON object: case id -> seed id -> frozen upstream run directory",
    )
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--profile", type=Path, help="Matrix profile (default: the maintained default)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        rows = []
        from .task_inventory import CORE_TASK_DIRECTORIES
        paths = ([DEFAULT_TASK_ROOT / name for name in CORE_TASK_DIRECTORIES]
                 if args.core else sorted(DEFAULT_TASK_ROOT.iterdir()))
        for path in paths:
            if (path / "task.yaml").is_file():
                spec = load_task(path)
                rows.append({"id": spec.task_id, "title": spec.title, "kind": spec.kind,
                             "availability": spec.availability})
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if args.command == "prepare-input":
        result = prepare_benchmark_input(
            args.profile,
            open3dbench_root=args.open3dbench_root,
            output_dir=args.output_dir,
            python_executable=args.python,
            dreamplace_install=args.dreamplace_install,
            seed_run=args.seed_run,
            timeout_seconds=args.timeout,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate-input":
        result = validate_benchmark_input(args.input, task_id=args.task)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "render-matrix":
        from .matrix_profile import load_matrix_profile, render_matrix, task_settings
        profile = load_matrix_profile(args.profile)
        task_path = _task_path(args.task)
        task_directory = (task_path.parent if task_path.is_file() else task_path).name
        matrix = render_matrix(
            task_directory,
            json.loads(args.baseline_runs.read_text(encoding="utf-8")),
            args.output_dir,
            profile=profile,
        )
        print(json.dumps({
            "matrix": str(matrix),
            "profile": profile["id"],
            "qualified": False,
            "next": "qualify-matrix with these settings before evaluating candidates",
            "settings": task_settings(task_directory, profile),
        }, indent=2))
        return 0
    spec = load_task(_task_path(args.task))
    if args.command == "inspect":
        print(json.dumps({
            "id": spec.task_id,
            "title": spec.title,
            "description": spec.description,
            "kind": spec.kind,
            "availability": spec.availability,
            "initial_candidate": str(spec.initial_candidate),
            "prompt": str(spec.prompt),
            "execution": {"from": spec.start_stage, "to": spec.stop_stage, "upstream_stages": spec.upstream_stages},
            "allowed_config_paths": spec.allowed_config_paths,
            "allowed_source_paths": spec.allowed_source_paths,
            "priority_parameters": spec.priority_parameters,
            "benchmark_input": spec.benchmark_input_id,
            "contract": task_contract(spec),
        }, indent=2, ensure_ascii=False))
        return 0
    if args.command == "export-openevolve":
        if not args.open3dbench_root.expanduser().is_dir():
            raise FileNotFoundError(args.open3dbench_root)
        if args.benchmark_profile:
            profile = load_benchmark_input_profile(args.benchmark_profile)
            if spec.task_id not in profile.compatible_tasks:
                raise ValueError(
                    "benchmark profile {} is not compatible with task {}".format(
                        profile.profile_id, spec.task_id
                    )
                )
            spec = replace(
                spec,
                base_config=profile.config,
                benchmark_input_id=profile.profile_id,
            )
        bundle = export_openevolve_bundle(
            spec,
            load_candidate(spec.initial_candidate),
            args.open3dbench_root,
            args.output_dir,
        )
        print(json.dumps(bundle, indent=2, ensure_ascii=False))
        return 0
    matrix = None
    transport = None
    if getattr(args, "matrix", None):
        from .matrix import LocalEvaluationMatrix, qualify_matrix
        if args.matrix_runtime:
            from .matrix_docker import DockerMatrixTransport
            transport = DockerMatrixTransport(**json.loads(args.matrix_runtime.read_text()))
        if args.command == "qualify-matrix":
            result = qualify_matrix(args.matrix, spec.root, args.open3dbench_root, args.output_dir,
                                    transport=transport, workers=args.matrix_workers, timeout_seconds=args.timeout)
            print(json.dumps(result.to_dict(), indent=2))
            return 0 if result.metrics.get("runs_successfully") == 1 else 1
        if any((args.baseline_run, args.reference_run, args.base_config, args.benchmark_input,
                args.dry_run, args.create_routing_reference)):
            raise ValueError("Matrix replaces single-cell paths and requires full candidate evaluation")
        output = args.output_dir.expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError("Matrix evaluation requires a new empty output directory")
        matrix = LocalEvaluationMatrix(args.matrix, spec)
    elif getattr(args, "matrix_runtime", None) or getattr(args, "matrix_workers", 1) != 1:
        raise ValueError("Matrix runtime/workers require --matrix")
    if args.benchmark_input is not None and args.baseline_run is not None:
        raise ValueError("use --benchmark-input or legacy --baseline-run, not both")
    baseline_run = args.baseline_run
    if args.benchmark_input is not None:
        baseline_run = resolve_benchmark_run(args.benchmark_input, task_id=spec.task_id)
    candidate = args.candidate
    if spec.program_mode == "task_adapter" and (
        candidate is None or candidate.expanduser().is_file()
    ):
        if candidate is None:
            program = initial_program_text(spec, load_candidate(spec.initial_candidate))
        else:
            program = candidate.expanduser().read_text(encoding="utf-8")
        materialized = materialize_program(spec, program, args.open3dbench_root)
        candidate = args.output_dir.expanduser().resolve() / "input_candidate"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "candidate.json").write_text(
            json.dumps(materialized, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif candidate is None:
        candidate = spec.initial_candidate
    elif candidate.expanduser().is_file() and candidate.name == "candidate.json":
        candidate = candidate.expanduser().parent
    if matrix is not None:
        from .verifier import verify_candidate
        verify_candidate(spec, load_candidate(candidate), args.open3dbench_root)
        result = matrix.evaluate(candidate, args.open3dbench_root, args.output_dir / "matrix",
                                 transport=transport, workers=args.matrix_workers,
                                 timeout_seconds=args.timeout, compact=args.compact_artifacts)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.metrics.get("runs_successfully") == 1 else 1
    result = evaluate_task(
        spec.root,
        candidate,
        args.open3dbench_root,
        args.output_dir,
        baseline_run=baseline_run,
        reference_run=args.reference_run,
        base_config=args.base_config,
        dry_run=args.dry_run,
        keep_workspace=not (args.discard_workspace or args.compact_artifacts),
        keep_runtime=not (args.discard_runtime or args.compact_artifacts),
        compact_artifacts=args.compact_artifacts,
        create_routing_reference=args.create_routing_reference,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.metrics.get("runs_successfully") or result.metrics.get("dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
