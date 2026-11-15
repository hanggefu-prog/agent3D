"""Render the official case/seed matrix from organizer-owned frozen prefixes."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .controller import TASK_IDS
from .scoring import PUBLIC_SPLITS, SEEDS
from .halo_policy import halo_for
from .sampling import case_seed_ids


def render(*, baseline, split, input_index, output):
    if baseline not in TASK_IDS or split not in (*PUBLIC_SPLITS, "hidden"):
        raise ValueError("Unknown GNN baseline or public split")
    index = json.loads(Path(input_index).read_text())
    if split == "hidden":
        cases = sorted(index) if isinstance(index, dict) else []
        if len(cases) != 2 or set(cases) & set(PUBLIC_SPLITS["train"] + PUBLIC_SPLITS["validation"]):
            raise ValueError("Hidden matrix needs two private, distinct cases")
    else:
        cases = list(PUBLIC_SPLITS[split])
    if not isinstance(index, dict) or set(index) != set(cases):
        raise ValueError("Input index must contain exactly the selected cases")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    framework = Path(__file__).resolve().parents[1]
    task = framework / "local_tasks" / TASK_IDS[baseline]
    template = yaml.safe_load((task / "reference_config.yaml").read_text())
    cells = []
    for case in cases:
        design, pdk = case.rsplit("-", 1)
        required_seeds = case_seed_ids(baseline, case)
        if (not isinstance(index[case], dict) or not set(required_seeds) <= set(index[case])
                or set(index[case]) - {str(seed) for seed in SEEDS}):
            raise ValueError("Input index lacks a complete seed group: " + case)
        for seed_id in required_seeds:
            seed = int(seed_id)
            prefix = Path(index[case][str(seed)]).resolve(strict=True)
            if prefix.is_symlink() or not prefix.is_dir():
                raise ValueError("Frozen upstream run must be a real directory")
            for stage in ("00_input", "10_floorplan"):
                if not (prefix / stage / "stage_manifest.json").is_file():
                    raise ValueError("Frozen run lacks " + stage)
            config = dict(template)
            config["design"] = design
            config["pdk"] = pdk
            config["pins"] = {**template["pins"], "seed": 1}
            config["partition"] = {**template["partition"], "seed": seed}
            config["placement"] = {**template["placement"], "seed": seed}
            if split != "hidden":
                config["placement"]["macro_halo_um"] = halo_for(case, seed)
            config["routing"] = {**template["routing"], "seed": -1}
            config_path = output / "configs" / case / (str(seed) + ".yaml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            cells.append({"case": case, "seed": str(seed), "base_config": str(config_path.resolve()),
                          "baseline_run": str(prefix), "reference_run": None})
    matrix = {"schema_version": 1, "task": TASK_IDS[baseline],
              "cases": cases,
              "seeds": [{"id": str(seed), "io": 1, "partition": seed,
                         "placement": seed, "routing": -1} for seed in SEEDS],
              "cells": cells}
    if any(len(case_seed_ids(baseline, case)) != len(SEEDS) for case in cases):
        matrix["schema_version"] = 2
        matrix["case_seeds"] = {case: list(case_seed_ids(baseline, case)) for case in cases}
    destination = output / "matrix.yaml"
    destination.write_text(yaml.safe_dump(matrix, sort_keys=False))
    return destination
