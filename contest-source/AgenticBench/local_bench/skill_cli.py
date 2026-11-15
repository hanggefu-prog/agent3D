"""Inspect the code-defined benchmark Skill inventory."""

from __future__ import annotations

import argparse
import json

from .skill_catalog import load_skill_catalog, required_skills, skill_by_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openagentic3dbench-skill",
        description="List and inspect benchmark-owned analyzer, task, and infrastructure Skills.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List all registered Skills")
    inspect = commands.add_parser("inspect", help="Inspect one Skill and its dependencies")
    inspect.add_argument("skill")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    records = load_skill_catalog()
    if args.command == "list":
        print(json.dumps([record.to_dict() for record in records], indent=2, ensure_ascii=False))
        return 0
    record = skill_by_name(records, args.skill)
    value = record.to_dict()
    value["resolved_requirements"] = [row.to_dict() for row in required_skills(records, record.name)]
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
