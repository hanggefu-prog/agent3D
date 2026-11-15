"""Read-only diagnostics. No reruns, candidate rewards, or missing-value fill."""
import csv
from pathlib import Path


def assignment_comparison(run_dir, reference_run):
    if reference_run is None:
        return {}
    paths = [Path(root) / "20_partition/partition_assignment.csv"
             for root in (run_dir, reference_run)]
    if not all(path.is_file() for path in paths):
        return {}
    def read(path):
        rows = {}
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["inst_name", "die"]:
                raise ValueError("Assignment diagnostic requires canonical columns")
            for row in reader:
                name, die = row["inst_name"], row["die"]
                if not name or name in rows or die not in {"bottom", "top"}:
                    raise ValueError("Invalid assignment diagnostic record")
                rows[name] = die
        return rows
    candidate, reference = map(read, paths)
    if not candidate or candidate.keys() != reference.keys():
        raise ValueError("Assignment diagnostic instance identities differ")
    changed = sum(candidate[name] != reference[name] for name in candidate)
    return {"partition_assignment_changed_count": changed,
            "partition_assignment_changed_fraction": changed / len(candidate),
            "partition_assignment_identical": int(changed == 0)}


def placement_log_diagnostics(run_dir):
    # Reuse the exact GP row delimiters, but never coerce missing/NaN to zero.
    import math
    import re
    from .metrics import _WEIGHTED_HPWL, _OVERFLOW_VECTOR, _OVERFLOW_SCALAR, _MAX_DENSITY_VECTOR, _MAX_DENSITY_SCALAR
    path = Path(run_dir) / "30_placement/stages/02_global_place/run.log"
    if not path.is_file():
        return {}
    rows = []
    def vector(line, many, single):
        match = many.search(line)
        if match:
            return [float(x) for x in re.split(r"[\s,]+", match.group(1).strip()) if x]
        match = single.search(line)
        return [float(match.group(1))] if match else []
    for line in path.read_text(errors="replace").splitlines():
        hpwl = _WEIGHTED_HPWL.search(line)
        if not hpwl:
            continue
        try:
            overflow = vector(line, _OVERFLOW_VECTOR, _OVERFLOW_SCALAR)
            density = vector(line, _MAX_DENSITY_VECTOR, _MAX_DENSITY_SCALAR)
            h = float(hpwl.group(1))
        except ValueError:
            continue
        if overflow and density and all(math.isfinite(x) and x >= 0 for x in [h, *overflow, *density]):
            rows.append((h, max(overflow)))
    if not rows:
        return {}
    first, last = rows[0], rows[-1]
    result = {"placement_logged_rows": len(rows),
              "placement_first_logged_hpwl": first[0],
              "placement_first_logged_overflow_max": first[1],
              "placement_logged_overflow_change": last[1] - first[1]}
    if first[0] > 0:
        result["placement_logged_hpwl_relative_change"] = (last[0] - first[0]) / first[0]
    return result
