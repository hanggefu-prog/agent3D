"""Read-only OpenROAD validation against retained failed-reference databases.

Writes only new diagnostic artifacts; never edits the frozen run or source.
Run inside the pinned node image with source and node paths mounted.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from open3dbench.backend.three_d_opt import _platform_files, _run_openroad

source, reference, patch, output, platform, binary = map(Path, sys.argv[1:7])
materialize = sys.argv[7:] == ["--materialized"]
if sys.argv[7:] and not materialize:
    raise ValueError("unsupported diagnostic mode")
output.mkdir(parents=True, exist_ok=False)
target = output / "patched/open3dbench/backend/three_d_opt.py"
if materialize:
    from local_bench.workspace import isolated_clone, apply_candidate_patch
    isolated_clone(source, output / "patched")
    apply_candidate_patch(output / "patched", patch)
else:
    target.parent.mkdir(parents=True)
    shutil.copy2(source / "open3dbench/backend/three_d_opt.py", target)
    subprocess.run(["git", "apply", "--unsafe-paths", "--directory", str(output / "patched"), str(patch)], check=True)
node = next(n for n in ast.parse(target.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_quiet_base_port_queries")
namespace = {"re": re}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(target), "exec"), namespace)
guard = namespace[node.name]
report = {}
for die in ("bottom", "top"):
    original = reference / "run/60_opt/iterations/01/constraints" / (die + ".sdc")
    guarded = output / (die + ".sdc")
    guarded.write_text(guard(original.read_text()))
    odb = reference / "run/60_opt/iterations/01" / die / "repair_design.odb"
    _, libs = _platform_files(platform, die)
    script = "read_db {" + str(odb) + "}\n"
    script += "\n".join("read_liberty {" + str(lib) + "}" for lib in libs)
    script += "\nset_cmd_units -time ns\nputs \"RESET_PORT_COUNT=[llength [get_ports -quiet rst_ni]]\"\n"
    script += "read_sdc {" + str(guarded) + "}\nputs GUARD_SDC_VALID\n"
    report[die] = _run_openroad(script, {}, output / (die + ".log"), str(binary))
    log = (output / (die + ".log")).read_text()
    assert "GUARD_SDC_VALID" in log
    report[die]["reset_port_count"] = int(re.search(r"RESET_PORT_COUNT=(\d+)", log).group(1))
    report[die]["original_sdc_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
report["patched_source_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
report["workspace_materialized"] = materialize
report["scope"] = "die-local SDC only; original and assembly constraints untouched"
(output / "qualification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
