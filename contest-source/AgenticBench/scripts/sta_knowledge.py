#!/usr/bin/env python3
"""Inspect/export an explicit STA knowledge store without running a model or STA."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic3d.knowledge import KnowledgeStore, read_package, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("path", type=Path, help="knowledge directory or exported package.json")
    export = sub.add_parser("export")
    export.add_argument("run", type=Path)
    export.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        if args.path.is_dir():
            store = KnowledgeStore(args.path)
            print(canonical({"manifest": store.manifest, "active_records": len(store.records())}))
        else:
            package = read_package(args.path)
            print(canonical({"sha256": package["sha256"],
                             "training_cases": package["payload"]["training_cases"],
                             "records": len(package["payload"]["records"])}))
    else:
        summary = json.loads((args.run / "summary.json").read_text())
        if summary.get("protocol") != "knowledge-v2" or summary.get("final_sta_evaluations") != 5:
            raise ValueError("export requires a completed five-round knowledge-v2 training run")
        KnowledgeStore(args.run / "knowledge").export(args.destination)
        print("Exported training knowledge: " + str(args.destination.resolve()))


if __name__ == "__main__":
    main()
