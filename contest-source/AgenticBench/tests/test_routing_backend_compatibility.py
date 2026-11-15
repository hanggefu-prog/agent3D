"""Routing source adapters must preserve the current per-die timeout contract."""
import ast
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional
from unittest import mock

import pytest

from local_tasks.routing_source_common import grt_only_patches


def patched_runner():
    root = Path(__file__).resolve().parents[2]
    source = root / "open3dbench/backend/three_d_route.py"
    if not source.is_file():
        pytest.skip("consolidated Open3DBench checkout is required")
    transformed, _ = grt_only_patches(root)
    tree = ast.parse(transformed)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_openroad")
    written = []
    namespace = {
        "Path": Path, "Dict": dict, "Mapping": Mapping, "Optional": Optional,
        "time": time, "subprocess": subprocess, "os": os,
        "ThreeDRouteError": RuntimeError, "json": __import__("json"),
        "_audit_warnings": lambda *a, **kw: {"unexpected_count": 0},
        "_write_json": lambda path, payload: written.append((path, payload)),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_run_openroad"], written


def test_routing_adapter_keeps_timeout_and_injects_die_context(tmp_path):
    run, _ = patched_runner()
    with mock.patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=0)) as call:
        run(tmp_path / "route.tcl", tmp_path / "route.log", "openroad",
            extra_environment={
                "OPEN3D_ROUTING_DIE": "top",
                "OPEN3D_HBT_PORT_FILE": str(tmp_path / "hbt.tsv"),
            },
            timeout_seconds=7)
    assert call.call_args.kwargs["timeout"] == 7
    assert call.call_args.kwargs["env"]["OPEN3D_ROUTING_DIE"] == "top"
    assert call.call_args.kwargs["env"]["OPEN3D_HBT_PORT_FILE"] == str(tmp_path / "hbt.tsv")


def test_routing_adapter_still_records_and_raises_timeout(tmp_path):
    run, written = patched_runner()
    with mock.patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("openroad", 7)):
        with pytest.raises(RuntimeError, match="per-die timeout"):
            run(tmp_path / "route.tcl", tmp_path / "route.log", "openroad",
                extra_environment={
                    "OPEN3D_ROUTING_DIE": "bottom",
                    "OPEN3D_HBT_PORT_FILE": str(tmp_path / "hbt.tsv"),
                },
                timeout_seconds=7)
    assert len(written) == 2
    assert all(record["failure"] == "die-routing-timeout" for _, record in written)
    assert all(record["timeout_seconds"] == 7 for _, record in written)
