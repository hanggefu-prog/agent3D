"""Read-only inventory of existing OPT/CTS artifacts, not native qualification.

This deliberately cannot issue a qualification or dispatch a candidate. It
detects input-contract problems before a paid developer is asked to solve them.
"""
import csv
import math
from pathlib import Path

from .metrics import clock_skew
from .protocol import sha


def placement(path):
    result = {}
    for fields in csv.reader(Path(path).open(), delimiter="\t"):
        if fields[0] != "I":
            continue
        if len(fields) != 8:
            raise ValueError("Malformed placement dump")
        key = (fields[1], fields[2])
        if key in result:
            raise ValueError("Repeated physical cell")
        result[key] = fields[3:]
    if not result:
        raise ValueError("Empty placement dump")
    return result


def clock_table(path):
    """Keep primary/auxiliary clocks, active edges and min/max views separate."""
    result = {}
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = ["clock", "die", "pin", "active_edge", "arrival_max_rise_ns",
                  "arrival_max_fall_ns", "arrival_min_rise_ns", "arrival_min_fall_ns",
                  "clock_domain_count"]
        if reader.fieldnames != fields:
            raise ValueError("Unexpected per-domain clock table")
        for row in reader:
            if row["die"] not in {"bottom", "top"} or row["active_edge"] not in {"rise", "fall"}:
                raise ValueError("Clock has no usable physical sink / active edge")
            if row["clock_domain_count"] != "1":
                raise ValueError("Ambiguous multi-domain clock sink needs explicit STA support")
            for analysis in ("max", "min"):
                group = (row["clock"], row["active_edge"], analysis, "default")
                arrival = float(row["arrival_" + analysis + "_" + row["active_edge"] + "_ns"])
                if not math.isfinite(arrival):
                    raise ValueError("Missing/nonfinite clock arrival")
                values = result.setdefault("/".join(group), {})
                if row["pin"] in values:
                    raise ValueError("Repeated clock sink")
                values[row["pin"]] = arrival
    if not result:
        raise ValueError("Empty clock table")
    return result


def audit(run, case):
    run = Path(run).resolve(strict=True)
    before_path = run / "60_opt/assembly/placement.tsv"
    after_path = run / "70_cts/assembly/placement.tsv"
    clock_path = run / "70_cts/assembly/passive/clock_sinks_by_clock.tsv"
    before, after = placement(before_path), placement(after_path)
    common = set(before) & set(after)
    moved = sorted("/".join(k) for k in common if before[k][1:4] != after[k][1:4])
    remastered = sorted("/".join(k) for k in common if before[k][0] != after[k][0])
    clocks = clock_table(clock_path)
    return {"kind": "engineering-input-audit-v1", "case": case, "run": str(run),
        "physical_qualification": False, "candidate_finals": 0,
        "opt_original_cell_count": len(before), "cts_cell_count": len(after),
        "original_cells_moved": len(moved), "moved_examples": moved[:12],
        "original_cells_remastered": len(remastered), "remastered_examples": remastered[:12],
        "original_cells_missing": sorted("/".join(k) for k in set(before) - set(after)),
        "observed_clock_groups": list(clocks),
        "observed_clock_skew_ns": clock_skew(clocks, {k: list(v) for k, v in clocks.items()}),
        "hashes": {str(p.relative_to(run)): sha(p) for p in (before_path, after_path, clock_path)},
        "note": "Historical artifacts inspected only; source/runtime/STA replay and new-task legality are not qualified."}
