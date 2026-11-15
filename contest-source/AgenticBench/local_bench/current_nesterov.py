"""Explicit current-source profile for the existing Nesterov local task.

The dated profiles and their receipts remain separate. This profile requires a
fresh initial-program reference from the same source, image and fixed input.
"""
import os
from pathlib import Path
import re

import yaml

from .reference_latest import require

TASK = "placement_dynamic_parameters_reference_20260923"
PROFILE = "current-source-nesterov-v1"
RECIPE = "../AgenticBench/local_bench/reference_configs/nesterov_current.json"


def selected(spec, base_config):
    if spec.task_id != TASK:
        return False
    config = yaml.safe_load(Path(base_config or spec.base_config).read_text())
    return config.get("placement", {}).get("config") == RECIPE


def contract(base_config, seed_schedule=None):
    config = yaml.safe_load(Path(base_config).read_text())
    template = yaml.safe_load((Path(__file__).parent / "reference_configs/nesterov_current.yaml").read_text())
    from open3dbench.benchmarks import get_design
    get_design(config.get("design"))
    template["design"] = config["design"]
    if seed_schedule is not None:
        from .seeds import validate_schedule
        seeds = validate_schedule(seed_schedule)
        for stage, key in (("pins", "io"), ("partition", "partition"),
                           ("placement", "placement"), ("routing", "routing")):
            template[stage]["seed"] = seeds[key]
    require(config == template, "Current Nesterov recipe differs from the fixed task profile")
    image = os.environ.get("OPEN3DBENCH_IMAGE_ID", "")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", image) is not None,
            "Current Nesterov profile requires a pinned installed runtime")
    return {"profile": PROFILE, "image": image, "config": config, "deterministic_flag": 1}


def admit(spec, candidate, source, baseline_run, reference_run, base_config, *, seed_schedule=None):
    from .scoped_reference import admit_reference
    context = contract(base_config, seed_schedule)
    if reference_run is not None:
        from .reference_placement_evidence import validate_reference_placement_evidence
        validate_reference_placement_evidence(reference_run, TASK, profile=PROFILE,
            placement_seed=seed_schedule["placement"] if seed_schedule else 1000)
    return admit_reference(spec, candidate, source, baseline_run, reference_run,
                           execution_contract=context)
