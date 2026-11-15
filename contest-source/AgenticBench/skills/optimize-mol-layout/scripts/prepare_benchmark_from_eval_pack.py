#!/usr/bin/env python3
"""Reconstruct a Place-MoL placement benchmark from an evaluation DEF."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_block_masters(lef_dir: Path) -> dict[str, float]:
    """Return BLOCK-class LEF masters and their areas in square microns."""
    masters: dict[str, float] = {}
    for lef_path in sorted(lef_dir.glob("*.lef")):
        current = None
        block = False
        area = None
        for raw_line in lef_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line.startswith("MACRO "):
                current = line.split(maxsplit=1)[1]
                block = False
                area = None
            elif current and line.startswith("CLASS BLOCK"):
                block = True
            elif current and line.startswith("SIZE "):
                match = re.match(r"SIZE\s+([0-9.eE+-]+)\s+BY\s+([0-9.eE+-]+)", line)
                if match:
                    area = float(match.group(1)) * float(match.group(2))
            elif current and line == f"END {current}":
                if block and area is not None:
                    masters[current] = area
                current = None
    if not masters:
        raise ValueError("platform LEFs contain no CLASS BLOCK macro masters")
    if max(masters.values()) <= 10.0:
        raise ValueError(
            "platform macro LEFs appear to contain post-split shrunk geometry; "
            "Place-MoL reconstruction requires the unshrunk platform"
        )
    return masters


def normalize_def(source: Path, destination: Path, block_masters: set[str]) -> dict:
    in_components = False
    component_count = 0
    suffix_count = 0
    fixed_to_placed = 0
    saw_nets = False
    upper_macro_names = []
    bottom_macro_names = []
    with source.open("r", encoding="utf-8", errors="replace") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            stripped = line.lstrip()
            if stripped.startswith("COMPONENTS "):
                in_components = True
            elif stripped.startswith("END COMPONENTS"):
                in_components = False
            elif stripped.startswith("NETS "):
                saw_nets = True

            if in_components and stripped.startswith("- "):
                ending = "\n" if line.endswith("\n") else ""
                body = line[:-1] if ending else line
                match = re.match(r"^(\s*-\s+\S+\s+)(\S+)(.*)$", body)
                if match:
                    instance = re.match(r"^\s*-\s+(\S+)", body).group(1)
                    master = match.group(2)
                    normalized = re.sub(r"_(?:upper|bottom)$", "", master)
                    suffix_match = re.search(r"_(upper|bottom)$", master)
                    if normalized in block_masters and suffix_match:
                        if suffix_match.group(1) == "upper":
                            upper_macro_names.append(instance)
                        else:
                            bottom_macro_names.append(instance)
                    if normalized != master:
                        suffix_count += 1
                    line = f"{match.group(1)}{normalized}{match.group(3)}{ending}"
                    component_count += 1
            elif in_components and "+ FIXED" in line:
                line = line.replace("+ FIXED", "+ PLACED", 1)
                fixed_to_placed += 1
            writer.write(line)
    if not saw_nets or component_count == 0 or suffix_count == 0:
        raise ValueError("input is not a suffixed, net-bearing Place-MoL evaluation DEF")
    return {
        "component_count": component_count,
        "master_suffixes_removed": suffix_count,
        "fixed_components_made_movable": fixed_to_placed,
        "nets_section_present": saw_nets,
        "frozen_partition": {
            "upper_die_macro_names": upper_macro_names,
            "bottom_die_macro_names": bottom_macro_names,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-def", required=True, type=Path)
    parser.add_argument("--platform", required=True, type=Path)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    for subdir in ("lef", "lib"):
        if not (args.platform / subdir).is_dir():
            parser.error(f"missing platform directory: {args.platform / subdir}")

    block_masters = read_block_masters(args.platform / "lef")
    design_dir = args.output / "or_3D" / args.design
    design_dir.mkdir(parents=True)
    shutil.copytree(args.platform / "lef", args.output / "nangate45" / "lef")
    shutil.copytree(args.platform / "lib", args.output / "nangate45" / "lib")
    destination = design_dir / "2_2_floorplan_io.def"
    stats = normalize_def(args.eval_def, destination, set(block_masters))
    manifest = {
        "kind": "reconstructed-place-mol-benchmark",
        "design": args.design,
        "source_def": str(args.eval_def.resolve()),
        "source_def_sha256": sha256(args.eval_def),
        "output_def": str(destination.resolve()),
        "output_def_sha256": sha256(destination),
        "block_master_count": len(block_masters),
        "macro_lef_max_area": max(block_masters.values()),
        **stats,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
