"""Inspect, qualify and report the single-submission engineering task suite."""
import argparse
import json
from pathlib import Path

from .tasks import TASK_IDS, load_task, read_json
from .metrics import aggregate
from .protocol import qualify
from .native_cli import COMMANDS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    native = sub.add_parser("native", help="Run a protected adapter; no implicit dispatch")
    native.add_argument("adapter", choices=COMMANDS)
    native.add_argument("arguments", nargs=argparse.REMAINDER)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("task", choices=TASK_IDS)
    check = sub.add_parser("check-qualification")
    check.add_argument("task", choices=TASK_IDS)
    check.add_argument("case")
    check.add_argument("receipt", type=Path)
    report = sub.add_parser("report")
    report.add_argument("task", choices=TASK_IDS)
    report.add_argument("run", type=Path)
    audit_input = sub.add_parser("audit-input")
    audit_input.add_argument("case")
    audit_input.add_argument("run", type=Path)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("campaign", type=Path)
    preflight_parser.add_argument("qualification_root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "native":
        from .native_cli import launch
        return launch(args.adapter, args.arguments)
    if args.command == "list":
        result = [{**load_task(name).data, "task_digest": load_task(name).digest(),
                   "physical_availability": "requires_case_matched_native_qualification"}
                  for name in TASK_IDS]
    elif args.command == "inspect":
        task = load_task(args.task)
        result = {**task.data, "task_digest": task.digest(),
                  "instruction": task.instruction, "judge_instruction": task.judge_instruction}
    elif args.command == "check-qualification":
        result = qualify(load_task(args.task), args.case, args.receipt)
    elif args.command == "audit-input":
        from .input_audit import audit
        result = audit(args.run, args.case)
    elif args.command == "preflight":
        from .campaign import load_campaign, preflight
        result = preflight(load_campaign(args.campaign), args.qualification_root)
    else:
        task = load_task(args.task)
        root = args.run
        if (root / "ledger").is_dir():
            if (root / "submission.json").exists():
                raise ValueError("Ambiguous ledger layout")
            root = root / "ledger"
        seal = read_json(root / "submission.json")
        if seal["task_digest"] != task.digest():
            raise ValueError("Report task differs from the sealed submission")
        outer_seal = args.run / "sealed.json"
        if outer_seal.exists():
            outer = read_json(outer_seal)
            if (outer.get("submission_sha256") != seal["submission_sha256"] or
                    outer.get("task_digest") != task.digest()):
                raise ValueError("Queue seal differs from the final ledger")
        records = []
        for case in task.data["cases"]:
            folder = root / case
            path = folder / "result.json"
            if not path.exists():
                path = folder / "attempt.json"
            if path.exists():
                row = read_json(path)
                if (row.get("case") != case or row.get("task_digest") != task.digest()
                        or row.get("submission_sha256") != seal["submission_sha256"]):
                    raise ValueError("Mismatched case evidence")
                records.append(row)
        result = aggregate(task, records)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if args.command == "preflight" and not result["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
