"""CLI for evaluator-owned candidate replay profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Collection, Optional

from .cli import main as task_cli_main
from .model import load_task, resolve_task_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = PROJECT_ROOT / "evaluation_profiles"


def _profile_path(value: str, allowed: Optional[Collection[str]]) -> Path:
    profile = resolve_task_path(PROFILE_ROOT, value)
    if allowed is not None and profile.name not in allowed:
        raise FileNotFoundError("profile is not available through this specialized CLI: {}".format(value))
    return profile


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List evaluator-owned profiles")
    inspect = commands.add_parser("inspect", help="Inspect one evaluation profile")
    inspect.add_argument("profile")
    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate an already selected candidate with one profile",
        add_help=False,
    )
    evaluate.add_argument("profile")
    evaluate.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def run_cli(
    argv=None,
    *,
    allowed_profiles: Optional[Collection[str]] = None,
    prog: str = "openagentic3dbench-profile",
    description: str = (
        "Run evaluator-owned candidate replay profiles. Profiles are not optimization "
        "tasks and are never exported to an evolution backend."
    ),
) -> int:
    args = build_parser(prog, description).parse_args(argv)
    allowed = frozenset(allowed_profiles) if allowed_profiles is not None else None
    if args.command == "list":
        rows = []
        for path in sorted(PROFILE_ROOT.iterdir()):
            if not (path / "profile.yaml").is_file() or (
                allowed is not None and path.name not in allowed
            ):
                continue
            spec = load_task(path)
            rows.append(
                {
                    "profile": path.name,
                    "id": spec.task_id,
                    "title": spec.title,
                    "kind": spec.kind,
                }
            )
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    profile = _profile_path(args.profile, allowed)
    spec = load_task(profile)
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "profile": profile.name,
                    "id": spec.task_id,
                    "title": spec.title,
                    "description": spec.description,
                    "kind": spec.kind,
                    "execution": {
                        "from": spec.start_stage,
                        "to": spec.stop_stage,
                        "upstream_stages": spec.upstream_stages,
                    },
                    "benchmark_input": spec.benchmark_input_id,
                    "allowed_config_paths": spec.allowed_config_paths,
                    "allowed_source_paths": spec.allowed_source_paths,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    return task_cli_main(["evaluate", str(profile), *args.arguments])


def main(argv=None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
