"""Audit self-reported GRT overflow against an independent guide-based estimate.

Usage examples::

    python AgenticBench/scripts/audit_grt_overflow.py RUN_ROOT [RUN_ROOT ...]
    python AgenticBench/scripts/audit_grt_overflow.py OUT/grt --output audit.json
    python AgenticBench/scripts/audit_grt_overflow.py CAND/grt --reference REF_AUDIT.json

Each PATH may be a GRT directory (``bottom/`` + ``top/``), one die
directory, a run root containing ``grt/``, or any tree; nested GRT
directories are discovered.  Exit status is 1 when any target is
inconsistent or failed closed (unless ``--no-fail``), 2 when nothing is found.
"""
import argparse
import json
import sys
from pathlib import Path

FRAMEWORK = Path(__file__).resolve().parents[1]
for _path in (FRAMEWORK, FRAMEWORK.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from grt_bench.common import write  # noqa: E402
from grt_bench.independent_overflow import DEFAULT_RULE, audit_grt, discover, load_reference  # noqa: E402


def _directions(text):
    result = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        name, _, direction = item.partition("=")
        if not name or not direction:
            raise argparse.ArgumentTypeError("expected LAYER=H|V[,LAYER=H|V...]")
        result[name] = direction
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="GRT dirs, die dirs or run roots")
    parser.add_argument("--reference", help="reference receipt JSON (applies the reference rule to every target)")
    parser.add_argument("--output", type=Path, help="write the combined JSON report here (never overwritten)")
    parser.add_argument("--rel-tolerance", type=float, default=DEFAULT_RULE["rel_tolerance"])
    parser.add_argument("--abs-tolerance", type=float, default=DEFAULT_RULE["abs_tolerance"])
    parser.add_argument("--reference-rel-tolerance", type=float, default=DEFAULT_RULE["reference_rel_tolerance"])
    parser.add_argument("--reference-abs-tolerance", type=float, default=DEFAULT_RULE["reference_abs_tolerance"])
    parser.add_argument("--require-positive-estimate", action="store_true")
    parser.add_argument("--platform-dir", type=Path, help="platform whose lef/*tech*.lef gives layer directions")
    parser.add_argument("--tech-lef", type=Path, help="explicit tech LEF")
    parser.add_argument("--layer-directions", type=_directions, help="explicit LAYER=H|V list")
    parser.add_argument("--gcell-size", type=int, help="explicit GCell size in DBU (default: OpenDB rule)")
    parser.add_argument("--track-model", choices=("average", "exact"), default="average")
    parser.add_argument("--no-manifest", action="store_true", help="skip grt_manifest.json hash checks")
    parser.add_argument("--no-fail", action="store_true", help="always exit 0 after reporting")
    parser.add_argument("--json", action="store_true", help="print the JSON report instead of a table")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rule = {"rel_tolerance": args.rel_tolerance, "abs_tolerance": args.abs_tolerance,
            "reference_rel_tolerance": args.reference_rel_tolerance,
            "reference_abs_tolerance": args.reference_abs_tolerance,
            "require_positive_estimate": args.require_positive_estimate}
    reference = load_reference(args.reference) if args.reference else None
    targets = []
    for path in args.paths:
        found = discover(path) if path.is_dir() else []
        targets.extend(found or [path])
    receipts = [audit_grt(target, rule=rule, reference=reference, platform=args.platform_dir,
                          tech_lef=args.tech_lef, layer_directions=args.layer_directions,
                          gcell_size=args.gcell_size, track_model=args.track_model,
                          require_manifest=not args.no_manifest) for target in targets]
    inconsistent = [item["grt_dir"] for item in receipts if not item["consistent"]]
    report = {"schema_version": 1, "targets": receipts,
              "summary": {"targets": len(receipts), "consistent": len(receipts) - len(inconsistent),
                          "inconsistent": inconsistent}}
    if args.output:
        write(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("{:<10} {:>10} {:>10}  {}".format("status", "reported", "estimate", "grt_dir"))
        for item in receipts:
            status = "ok" if item["consistent"] else ("error" if item["errors"] else "MISMATCH")
            print("{:<10} {:>10} {:>10}  {}".format(status, str(item["reported_overflow"]),
                                                    str(item["estimated_overflow"]), item["grt_dir"]))
            for error in item["errors"]:
                print("    {}: {}".format(error["scope"], error["error"]))
            for check in item["checks"]:
                if not check["passed"]:
                    print("    {} {} failed: reported {} estimate {} limit {}".format(
                        check["scope"], check["check"], check["reported"], check["estimate"], check.get("limit")))
        print("{}/{} consistent".format(report["summary"]["consistent"], len(receipts)))
    if not receipts:
        return 2
    return 0 if args.no_fail or not inconsistent else 1


if __name__ == "__main__":
    sys.exit(main())
