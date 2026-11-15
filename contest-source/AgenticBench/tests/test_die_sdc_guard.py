"""Deployment regression: absent local ports must not erase real exceptions."""
import ast
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guard(tmp_path):
    source = ROOT / "tmp/sta_codesign_source/open3dbench/backend/three_d_opt.py"
    if not source.exists():
        pytest.skip("exact STA source fixture not available")
    target = tmp_path / "open3dbench/backend/three_d_opt.py"
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    subprocess.run(["git", "apply", "--unsafe-paths", "--directory", str(tmp_path),
                    str(ROOT / "deploy/compute_pool/guard_absent_die_false_path.patch")], check=True)
    node = next(n for n in ast.parse(target.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_quiet_base_port_queries")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(target), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("present", [True, False])
def test_literal_exception_preserved_or_absent_noop(guard, present):
    sdc = guard("set_false_path -from [get_ports -quiet rst_ni]\n")
    script = (
        "set called 0\n"
        "proc get_ports {args} { return " + ("rst_ni" if present else "{}") + " }\n"
        "proc set_false_path {args} { global called; incr called; "
        "if {[lindex $args 1] ne {rst_ni}} {error {invalid exception}} }\n"
        + sdc + "\nputs $called\n"
    )
    result = subprocess.run(["tclsh"], input=script, text=True, capture_output=True, check=True)
    assert result.stderr == ""
    assert result.stdout.strip() == str(int(present))


def test_complex_constraints_are_not_relaxed(guard):
    for line in [
        "set_false_path -from [get_ports -quiet {rst_ni test_i}]",
        "set_false_path -from [get_ports -quiet rst_ni] -to [get_ports -quiet out]",
        "set_max_delay 4.0 -to [all_registers -data_pins -level_sensitive]",
        "set_min_delay 0.0 -to [all_registers -data_pins -level_sensitive]",
    ]:
        assert guard(line) == line
    assert guard("set_false_path -from [get_ports rst_ni]").startswith("if {[llength")
