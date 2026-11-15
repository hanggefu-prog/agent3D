"""Byte-identity rule for evaluator wrapper hunks in routing source patches.

The routing source tasks list ``open3dbench/backend/three_d_route.py`` and
``open3dbench/runner.py`` in ``allowed_source_paths`` only because the trusted
task adapter must expose die/HBT context to the evolved FastRoute hook.  Those
two files are evaluator wrappers, not search space.  A raw ``candidate.json``
(CLI, search API, OpenEvolve ``candidate.json`` programs, matrix cells) can
bypass the adapter, so the verifier independently re-derives the wrapper diff
from the same Open3DBench checkout and requires every wrapper hunk in the
candidate patch to be byte-identical to it.  Omitting a wrapper section is
allowed (the wrapper then stays native); any other wrapper content is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .model import ROUTING_SOURCE_TASK_KINDS, TaskSpec
from .patch_paths import split_patch_sections


ROUTE_BACKEND = "open3dbench/backend/three_d_route.py"
RUNNER = "open3dbench/runner.py"
WRAPPER_PATHS = (ROUTE_BACKEND, RUNNER)

def trusted_wrapper_diffs(open3dbench_root: Path) -> Dict[str, str]:
    """Re-derive the adapter's exact wrapper diffs from one checkout."""

    from local_bench.program_safety import read_repo_file, unified_diff
    from local_tasks.routing_source_common import grt_only_patches

    root = Path(open3dbench_root).expanduser().resolve()
    route_after, runner_after = grt_only_patches(root)
    return {
        ROUTE_BACKEND: unified_diff(
            ROUTE_BACKEND, read_repo_file(root, ROUTE_BACKEND), route_after
        ),
        RUNNER: unified_diff(RUNNER, read_repo_file(root, RUNNER), runner_after),
    }


def applies(spec: TaskSpec) -> bool:
    return spec.kind in ROUTING_SOURCE_TASK_KINDS and any(
        path in spec.allowed_source_paths for path in WRAPPER_PATHS
    )


def verify_wrapper_hunks(
    spec: TaskSpec, patch_text: str, open3dbench_root: Optional[Path]
) -> Dict[str, str]:
    """Reject wrapper sections that differ from trusted adapter output.

    Returns ``{wrapper path: "absent" | "trusted"}`` for audit artifacts.
    """

    if not applies(spec) or not patch_text.strip():
        return {}
    sections = split_patch_sections(patch_text)
    wrapper_sections: Dict[str, List[str]] = {}
    for path, text in sections:
        if path in WRAPPER_PATHS:
            wrapper_sections.setdefault(path, []).append(text)
    if not wrapper_sections:
        return {path: "absent" for path in WRAPPER_PATHS}
    if open3dbench_root is None:
        # Callers that cannot re-derive the trusted wrapper must not accept it.
        raise ValueError(
            "routing wrapper hunks require the Open3DBench checkout to verify "
            "byte identity with the trusted adapter output"
        )
    expected = trusted_wrapper_diffs(open3dbench_root)
    status = {}
    for path in WRAPPER_PATHS:
        found = wrapper_sections.get(path)
        if not found:
            status[path] = "absent"
            continue
        if len(found) != 1 or not expected[path] or found[0] != expected[path]:
            raise ValueError(
                "source patch hunk for evaluator wrapper {} is not byte-identical to "
                "the trusted task-adapter output; wrapper files are not search space".format(path)
            )
        status[path] = "trusted"
    return status
