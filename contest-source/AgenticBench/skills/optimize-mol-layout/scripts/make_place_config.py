#!/usr/bin/env python3
"""Create an immutable Place-MoL config variant using dotted JSON keys."""

import argparse
import json
from pathlib import Path


ALLOWED_NEW_KEYS = {
    "partition_params.frozen_partition_file",
    "dreamplace_overrides.target_density",
    "dreamplace_overrides.density_weight",
    "dreamplace_overrides.gamma",
    "dreamplace_overrides.stop_overflow",
    "dreamplace_overrides.gp_noise_ratio",
    "dreamplace_overrides.global_place_stages",
    "dreamplace_overrides.num_bins_x",
    "dreamplace_overrides.num_bins_y",
    "final_cell_dreamplace_overrides.target_density",
    "final_cell_dreamplace_overrides.density_weight",
    "final_cell_dreamplace_overrides.gamma",
    "final_cell_dreamplace_overrides.stop_overflow",
    "final_cell_dreamplace_overrides.gp_noise_ratio",
    "final_cell_dreamplace_overrides.global_place_stages",
    "final_cell_dreamplace_overrides.num_bins_x",
    "final_cell_dreamplace_overrides.num_bins_y",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_VALUE")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    data = json.loads(args.base.read_text(encoding="utf-8"))
    for assignment in args.set:
        if "=" not in assignment:
            parser.error(f"invalid --set value: {assignment!r}")
        dotted_key, encoded_value = assignment.split("=", 1)
        keys = dotted_key.split(".")
        cursor = data
        for key in keys[:-1]:
            if key not in cursor and dotted_key in ALLOWED_NEW_KEYS:
                cursor[key] = {}
            if key not in cursor or not isinstance(cursor[key], dict):
                parser.error(f"unknown object path: {dotted_key!r}")
            cursor = cursor[key]
        if keys[-1] not in cursor and dotted_key not in ALLOWED_NEW_KEYS:
            parser.error(f"unknown config key: {dotted_key!r}")
        try:
            cursor[keys[-1]] = json.loads(encoded_value)
        except json.JSONDecodeError as exc:
            parser.error(f"value for {dotted_key!r} is not JSON: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
