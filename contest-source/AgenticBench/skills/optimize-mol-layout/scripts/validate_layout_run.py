#!/usr/bin/env python3
from __future__ import annotations
"""Validate a completed Place-MoL layout run without DREAMPlace imports."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


SUFFIX_RE = re.compile(r"_(?:upper|bottom)$")


@dataclass(frozen=True)
class Component:
    name: str
    master: str
    status: str | None
    x: int | None
    y: int | None
    orient: str | None


@dataclass(frozen=True)
class DefData:
    units: int
    diearea: tuple[int, int, int, int]
    declared_components: int
    declared_nets: int
    components: dict[str, Component]


def parse_def(path: Path) -> DefData:
    units = None
    diearea = None
    declared_components = None
    declared_nets = None
    in_components = False
    pending: list[str] = []
    components: dict[str, Component] = {}

    def finish(record: str) -> None:
        match = re.match(r"\s*-\s+(\S+)\s+(\S+)", record)
        if not match:
            raise ValueError(f"malformed component in {path}: {record[:120]!r}")
        name, master = match.groups()
        placement = re.search(
            r"\+\s+(FIXED|PLACED|COVER|UNPLACED)"
            r"(?:\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+))?",
            record,
        )
        status = placement.group(1) if placement else None
        x = int(placement.group(2)) if placement and placement.group(2) else None
        y = int(placement.group(3)) if placement and placement.group(3) else None
        orient = placement.group(4) if placement and placement.group(4) else None
        if name in components:
            raise ValueError(f"duplicate component {name!r} in {path}")
        components[name] = Component(name, master, status, x, y, orient)

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = re.match(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;", line)
        if match:
            units = int(match.group(1))
        match = re.match(
            r"DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*"
            r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;",
            line,
        )
        if match:
            diearea = tuple(int(value) for value in match.groups())
        match = re.match(r"COMPONENTS\s+(\d+)\s*;", line)
        if match:
            declared_components = int(match.group(1))
            in_components = True
            continue
        if line == "END COMPONENTS":
            if pending:
                raise ValueError(f"unterminated component in {path}")
            in_components = False
            continue
        match = re.match(r"NETS\s+(\d+)\s*;", line)
        if match:
            declared_nets = int(match.group(1))

        if in_components:
            if line.startswith("- "):
                if pending:
                    raise ValueError(f"unterminated component before {line[:80]!r}")
                pending = [line]
            elif pending:
                pending.append(line)
            if pending and ";" in line:
                finish(" ".join(pending))
                pending = []

    if units is None or diearea is None or declared_components is None or declared_nets is None:
        raise ValueError(f"missing required DEF header in {path}")
    if len(components) != declared_components:
        raise ValueError(
            f"component header mismatch in {path}: {declared_components} != {len(components)}"
        )
    return DefData(units, diearea, declared_components, declared_nets, components)


def parse_lef_sizes(lef_dir: Path) -> dict[str, tuple[float, float]]:
    sizes: dict[str, tuple[float, float]] = {}
    for path in sorted(lef_dir.glob("*.lef")):
        current = None
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("MACRO "):
                current = line.split(maxsplit=1)[1]
            elif current and line.startswith("SIZE "):
                match = re.match(r"SIZE\s+([0-9.eE+-]+)\s+BY\s+([0-9.eE+-]+)", line)
                if match:
                    sizes[current] = (float(match.group(1)), float(match.group(2)))
            elif current and line == f"END {current}":
                current = None
    if not sizes:
        raise ValueError(f"no macro sizes found under {lef_dir}")
    return sizes


def normalized_master(master: str) -> str:
    return SUFFIX_RE.sub("", master)


def rectangle(
    component: Component,
    units: int,
    sizes: dict[str, tuple[float, float]],
) -> tuple[int, int, int, int]:
    if component.x is None or component.y is None or component.orient is None:
        raise ValueError(f"component {component.name!r} has no placed coordinates")
    master = normalized_master(component.master)
    if master not in sizes:
        raise ValueError(f"missing LEF size for master {master!r}")
    width, height = sizes[master]
    if component.orient in {"E", "W", "FE", "FW"}:
        width, height = height, width
    width_dbu = int(math.ceil(width * units - 1e-9))
    height_dbu = int(math.ceil(height * units - 1e-9))
    return component.x, component.y, component.x + width_dbu, component.y + height_dbu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-def", required=True, type=Path)
    parser.add_argument("--macro-reference-def", required=True, type=Path)
    parser.add_argument("--cell-placed-def", required=True, type=Path)
    parser.add_argument("--final-def", required=True, type=Path)
    parser.add_argument("--lef-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    input_data = parse_def(args.input_def)
    macro_reference = parse_def(args.macro_reference_def)
    cell_placed = parse_def(args.cell_placed_def)
    final_data = parse_def(args.final_def)
    sizes = parse_lef_sizes(args.lef_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frozen = manifest["frozen_partition"]
    upper = set(frozen["upper_die_macro_names"])
    bottom = set(frozen["bottom_die_macro_names"])
    macro_names = upper | bottom

    input_masters = {
        name: normalized_master(component.master)
        for name, component in input_data.components.items()
    }
    final_masters = {
        name: normalized_master(component.master)
        for name, component in final_data.components.items()
    }
    identity_ok = input_masters == final_masters
    counts_ok = (
        input_data.declared_components == final_data.declared_components
        and input_data.declared_nets == final_data.declared_nets
    )

    bounds_violations = []
    xl, yl, xh, yh = final_data.diearea
    for component in final_data.components.values():
        try:
            left, bottom_y, right, top = rectangle(component, final_data.units, sizes)
        except ValueError as exc:
            bounds_violations.append(str(exc))
            continue
        if left < xl or bottom_y < yl or right > xh or top > yh:
            bounds_violations.append(
                f"{component.name}: ({left},{bottom_y},{right},{top}) outside {final_data.diearea}"
            )

    preserved_violations = []
    for name in sorted(macro_names):
        before = macro_reference.components.get(name)
        after = cell_placed.components.get(name)
        if before is None or after is None:
            preserved_violations.append(f"missing macro {name}")
        elif (before.x, before.y, before.orient) != (after.x, after.y, after.orient):
            preserved_violations.append(
                f"{name}: {(before.x, before.y, before.orient)} != "
                f"{(after.x, after.y, after.orient)}"
            )

    overlaps = []
    for die, names in (("upper", upper), ("bottom", bottom)):
        ordered = sorted(names)
        for index, left_name in enumerate(ordered):
            left_rect = rectangle(final_data.components[left_name], final_data.units, sizes)
            for right_name in ordered[index + 1 :]:
                right_rect = rectangle(final_data.components[right_name], final_data.units, sizes)
                if (
                    max(left_rect[0], right_rect[0]) < min(left_rect[2], right_rect[2])
                    and max(left_rect[1], right_rect[1]) < min(left_rect[3], right_rect[3])
                ):
                    overlaps.append(f"{die}: {left_name} overlaps {right_name}")

    suffix_violations = []
    for name, component in final_data.components.items():
        expected = "upper" if name in upper else "bottom"
        if not component.master.endswith(f"_{expected}"):
            suffix_violations.append(
                f"{name}: expected _{expected}, got {component.master}"
            )

    checks = {
        "component_and_net_counts_match": counts_ok,
        "component_identity_and_masters_match": identity_ok,
        "all_components_inside_diearea": not bounds_violations,
        "macro_coordinates_preserved_during_cell_placement": not preserved_violations,
        "no_same_die_macro_overlap": not overlaps,
        "suffixed_def_partition_matches_manifest": not suffix_violations,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "components": final_data.declared_components,
            "nets": final_data.declared_nets,
            "upper_macros": len(upper),
            "bottom_macros": len(bottom),
        },
        "violations": {
            "bounds": bounds_violations[:100],
            "macro_preservation": preserved_violations,
            "overlap": overlaps,
            "suffix": suffix_violations[:100],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
