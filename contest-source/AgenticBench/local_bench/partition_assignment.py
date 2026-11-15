"""Read and validate the canonical two-tier assignment used by reference audits."""

import csv
from pathlib import Path
from typing import Any, Dict, Mapping


def read_assignment(path: Path, raw: Mapping[str, Any]) -> Dict[str, str]:
    tiers = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["inst_name", "die"]:
            raise ValueError("partition requires canonical inst_name,die assignment")
        for row in reader:
            name, die = row["inst_name"], row["die"]
            if not name or name in tiers or die not in {"top", "bottom"}:
                raise ValueError("invalid or duplicate partition assignment")
            tiers[name] = die
    for key, actual in (("instance_count", len(tiers)),
                        ("top_count", sum(d == "top" for d in tiers.values())),
                        ("bottom_count", sum(d == "bottom" for d in tiers.values()))):
        if not tiers or raw.get(key) != actual:
            raise ValueError("partition assignment/count mismatch: " + key)
    return tiers
