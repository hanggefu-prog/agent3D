#!/usr/bin/env python3
"""Run the current local, engineering and post-GRT integration gate.

This selects maintained contracts explicitly. It does not hide failures from
the optional full historical suite or dispatch EDA/model evaluations.
"""
from pathlib import Path
import os
import subprocess
import sys


PATTERNS = (
    "test_openevolve_ask_tell.py", "test_local_matrix*.py", "test_search_api.py", "test_local_bench.py", "test_hierarchical.py", "test_local_task_smoke.py",
    "test_repository_boundary.py", "test_hbt_native.py", "test_hbt_policy_task.py",
    "test_current_task_inventory.py", "test_routing_backend_compatibility.py",
    "test_task_prompt_contracts.py", "test_latest*.py", "test_reference_latest*.py",
    "test_routing*.py", "test_qualified_routing_runtime.py", "test_placement*.py",
    "test_reference_local_placement.py", "test_reference_placement_evidence.py",
    "test_bin_timing_and_placement_components.py", "test_global_2way_timing_task.py",
    "test_tp_gnn*.py", "test_grt*.py", "test_engineering*.py", "engineering_native",
    "test_evolution_bridge.py", "test_openevolve_compiler_cache.py",
    "test_partition_research_definitions.py", "test_gnn*.py",
    "test_portable_launcher.py", "test_openroad_compiler_cache.py",
)


def main():
    framework = Path(__file__).resolve().parents[1]
    tests = framework / "tests"
    selected = set()
    for pattern in PATTERNS:
        matches = list(tests.glob(pattern))
        if not matches:
            raise RuntimeError("Maintained test group is missing: " + pattern)
        selected.update(str(path.relative_to(framework)) for path in matches)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(map(str, (
        framework, framework.parent, framework.parent / "Place-3D")))
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            *sorted(selected), *sys.argv[1:]],
                           cwd=framework, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
