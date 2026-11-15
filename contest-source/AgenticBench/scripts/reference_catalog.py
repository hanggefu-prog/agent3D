#!/usr/bin/env python3
"""Read-only producer qualification and append-only reference registration.

This command NEVER launches FRESH/EXTEND computation. It qualifies completed
outputs and publishes immutable records that subsequent ADOPT requests reuse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from local_bench import reference_catalog as catalog
from local_bench import reference_store as store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("qualify", help="Verify native evidence; no store write or EDA")
    inspect.add_argument("--producer", required=True, type=Path)
    register = sub.add_parser("register", help="Seal an independently qualified completed baseline")
    register.add_argument("--producer", required=True, type=Path)
    register.add_argument("--origin", choices=store.ORIGINS, required=True)
    register.add_argument("--store", required=True, type=Path)
    register.add_argument("--contract", required=True, type=Path,
                          help="Explicit task binding; no inferred compatibility or endpoint changes")
    adopt = sub.add_parser("adopt", help="Revalidate and adopt a sealed baseline without executing EDA")
    adopt.add_argument("--record", required=True, type=Path)
    adopt.add_argument("--contract", required=True, type=Path)
    adopt.add_argument("--output", required=True, type=Path)
    listing = sub.add_parser("list", help="List registrations, not a claim of fresh qualification")
    listing.add_argument("--store", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "qualify":
        result = catalog.qualify_producer(json.loads(args.producer.read_text()))
    elif args.action == "register":
        path = catalog.register_producer(args.store, json.loads(args.producer.read_text()),
                                        origin=args.origin, contract=json.loads(args.contract.read_text()))
        result = {"record": str(path), "reference_id": path.stem, "new_native_executions": 0}
    elif args.action == "adopt":
        result = catalog.adopt(args.record, contract=json.loads(args.contract.read_text()), output=args.output)
    else:
        result = {"status": "inventory-only-not-requalified", "records": []}
        for path in sorted((args.store / "records").glob("*.json")):
            row = json.loads(path.read_text())
            result["records"].append({"record": str(path.resolve()), "reference_id": row.get("reference_id"),
                "origin": row.get("origin"), "contract": row.get("contract"),
                "published_at": row.get("published_at")})
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
