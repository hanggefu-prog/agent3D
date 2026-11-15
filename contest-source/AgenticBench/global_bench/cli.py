"""Command-line entry point for workspace-oriented global benchmark trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from .impact import ImpactResolver
from .intervention import (
    GLOBAL_WORKSPACE_ANALYZER_SCHEMA,
    GlobalSearchContext,
    build_global_analyzer_prompt,
)
from .model import GlobalTaskSpec, load_global_task
from .runtime import DockerTrialRuntime
from .scheduler import GlobalTrialScheduler
from .submission import collect_submission
from .verifier import verify_submission_structure
from .workspace import TrialLayout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ROOT = PROJECT_ROOT / "global_tasks"
DEFAULT_SKILLS_ROOT = PROJECT_ROOT / "skills"


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _catalog(root: Path) -> Dict[str, GlobalTaskSpec]:
    task_root = root.expanduser().resolve()
    records = {}
    if not task_root.is_dir():
        raise ValueError("global task root is not a directory: {}".format(task_root))
    for path in sorted(task_root.iterdir()):
        if path.is_dir() and (path / "task.yaml").is_file():
            task = load_global_task(path)
            if task.task_id in records:
                raise ValueError("duplicate global task ID: {}".format(task.task_id))
            records[task.task_id] = task
    return records


def _task(arguments: argparse.Namespace) -> GlobalTaskSpec:
    records = _catalog(arguments.task_root)
    if arguments.task_id not in records:
        raise ValueError("unknown global task: {}".format(arguments.task_id))
    return records[arguments.task_id]


def _artifacts(values: Sequence[str], trial: TrialLayout) -> Dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("artifact must use NAME=PATH syntax")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = trial.root / path
        if name in result:
            raise ValueError("duplicate artifact: {}".format(name))
        result[name] = path
    return result


def _context(path: Path) -> GlobalSearchContext:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GlobalSearchContext(
        task_id=str(data["task_id"]),
        user_command=str(data["user_command"]),
        global_round=int(data["global_round"]),
        accepted_revision=str(data["accepted_revision"]),
        objectives=dict(data["objectives"]),
        hard_constraints=tuple(str(item) for item in data["hard_constraints"]),
        stage_artifacts=dict(data["stage_artifacts"]),
        observations=tuple(dict(item) for item in data["observations"]),
        remaining_budget=dict(data["remaining_budget"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list global task definitions")

    inspect = subparsers.add_parser("inspect", help="inspect a global task")
    inspect.add_argument("task_id")

    prepare = subparsers.add_parser("prepare", help="create one isolated trial worktree")
    prepare.add_argument("task_id")
    prepare.add_argument("--source-root", required=True, type=Path)
    prepare.add_argument("--trials-root", required=True, type=Path)
    prepare.add_argument("--revision")
    prepare.add_argument("--trial-id")
    prepare.add_argument("--user-command")

    release = subparsers.add_parser("release", help="release a trial concurrency slot")
    release.add_argument("--trial-dir", required=True, type=Path)
    release.add_argument("--trials-root", required=True, type=Path)
    release.set_defaults(no_task=True)

    impact = subparsers.add_parser("impact", help="resolve an adaptive evaluation plan")
    impact.add_argument("task_id")
    impact.add_argument("--changed-file", action="append", default=[])
    impact.add_argument("--claimed-stage", action="append", default=[])
    impact.add_argument("--policy", choices=("quick", "suffix", "final", "full"))

    schema = subparsers.add_parser("analyzer-schema", help="print the global Intervention schema")
    schema.set_defaults(no_task=True)

    prompt = subparsers.add_parser("analyzer-prompt", help="render a Global Analyzer prompt")
    prompt.add_argument("task_id")
    prompt.add_argument("--context", required=True, type=Path)

    runtime = subparsers.add_parser("runtime-plan", help="render or execute an isolated agent container")
    runtime.add_argument("task_id")
    runtime.add_argument("--trial-dir", required=True, type=Path)
    runtime.add_argument("--skills-root", type=Path, default=DEFAULT_SKILLS_ROOT)
    runtime.add_argument("--gpu", action="append", default=[])
    runtime.add_argument("--execute", action="store_true")
    runtime.add_argument("agent_command", nargs=argparse.REMAINDER)

    collect = subparsers.add_parser("collect", help="freeze a trial submission")
    collect.add_argument("task_id")
    collect.add_argument("--trial-dir", required=True, type=Path)
    collect.add_argument("--artifact", action="append", default=[])

    verify = subparsers.add_parser("verify", help="run trusted structural submission checks")
    verify.add_argument("task_id")
    verify.add_argument("--submission", required=True, type=Path)
    verify.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "list":
            _json(
                [
                    {
                        "id": task.task_id,
                        "title": task.title,
                        "default_policy": task.evaluation.default_policy,
                        "stages": list(task.evaluation.stage_ids()),
                    }
                    for task in _catalog(arguments.task_root).values()
                ]
            )
            return 0
        if arguments.command == "analyzer-schema":
            _json(GLOBAL_WORKSPACE_ANALYZER_SCHEMA)
            return 0
        if arguments.command == "release":
            layout = TrialLayout.from_path(arguments.trial_dir)
            GlobalTrialScheduler(arguments.trials_root).release(layout)
            _json({"trial_id": layout.trial_id, "released": True})
            return 0
        task = _task(arguments)
        if arguments.command == "inspect":
            _json(task.to_dict())
            return 0
        if arguments.command == "prepare":
            scheduled = GlobalTrialScheduler(arguments.trials_root).prepare(
                task,
                arguments.source_root,
                revision=arguments.revision,
                trial_id=arguments.trial_id,
                user_command=arguments.user_command,
            )
            _json(scheduled.to_dict())
            return 0
        if arguments.command == "impact":
            plan = ImpactResolver(task.evaluation).resolve(
                arguments.changed_file,
                claimed_stages=arguments.claimed_stage,
                policy=arguments.policy,
            )
            _json(plan.to_dict())
            return 0
        if arguments.command == "analyzer-prompt":
            context = _context(arguments.context)
            if context.task_id != task.task_id:
                raise ValueError("context belongs to a different task")
            print(build_global_analyzer_prompt(task, context))
            return 0
        if arguments.command == "runtime-plan":
            layout = TrialLayout.from_path(arguments.trial_dir)
            command = list(arguments.agent_command)
            if command and command[0] == "--":
                command.pop(0)
            plan = DockerTrialRuntime().plan(
                task,
                layout,
                command,
                skills_root=arguments.skills_root,
                assigned_gpus=arguments.gpu,
            )
            (layout.root / "runtime_plan.json").write_text(
                json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if arguments.execute:
                result = DockerTrialRuntime().run(
                    plan,
                    layout,
                    task.environment.walltime_seconds,
                    storage_limit_gb=task.environment.storage_gb,
                )
                _json(result.to_dict())
                return 0 if result.exit_code == 0 else 1
            _json(plan.to_dict())
            return 0
        if arguments.command == "collect":
            layout = TrialLayout.from_path(arguments.trial_dir)
            manifest = collect_submission(task, layout, _artifacts(arguments.artifact, layout))
            _json(manifest.to_dict())
            return 0
        if arguments.command == "verify":
            report = verify_submission_structure(task, arguments.submission)
            if arguments.output:
                output = arguments.output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            _json(report.to_dict())
            return 0 if report.passed else 2
        raise AssertionError("unhandled command")
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
