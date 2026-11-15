"""T4 harness-owned data-placement normalization and native legalization.

Only data/physical original instances are reset. Clock roles are proven from
frozen ODB + Liberty + SDC snapshots, not from names or candidate declarations.
This is one physical gate, not a replacement for equivalence or joint STA.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .native_snapshot import NATIVE, bind, cells
from .protocol import read_json, sha, digest, write_new, now, path_within

POLICY = {"id": "cts-harness-legalization-v1", "data_initialization": "frozen-post-opt",
          "retain_candidate_clock_locations": True, "max_displacement": [500, 500],
          "threads": 8, "pg_check": "VDD/VSS-exact", "candidate_editable": False}


def is_fixed(row):
    # CORE variants are standard cells. Macros/cover anchors and original fixed
    # cells are never released to the legalizer by candidate-controlled flags.
    return not row["master_type"].startswith("CORE") or row["status"] in {"FIRM", "LOCKED"}


def restoration(original, candidate):
    restore = []
    removed_clock = []
    for name, before in sorted(original.items()):
        if before["clock_cell"]:
            if name not in candidate:
                removed_clock.append(name)
            continue
        after = candidate.get(name)
        if after is None or before["master"] != after["master"]:
            raise ValueError("Original data/physical cell missing or resized: " + name)
        if is_fixed(before) and any(after[k] != before[k] for k in ("x_dbu", "y_dbu", "orient")):
            raise ValueError("Fixed macro/HBT moved before normalization: " + name)
        values = [name, before["master"], before["x_dbu"], before["y_dbu"],
                  before["orient"], before["status"], int(is_fixed(before))]
        if any("\t" in str(v) or "\n" in str(v) for v in values):
            raise ValueError("Unsafe native instance identity")
        restore.append("\t".join(map(str, values)))
    for name in set(candidate) - set(original):
        if not candidate[name]["clock_cell"]:
            raise ValueError("CTS added a non-clock cell: " + name)
    return "\n".join(restore) + "\n", removed_clock


def load_snapshot(folder):
    folder = Path(folder)
    dispatch = read_json(folder / "dispatch.json")
    result = read_json(folder / "result.json")
    if result.get("status") != "succeeded" or result.get("statistics", {}).get("floating_pg") != 0:
        raise ValueError("Unsuccessful/floating-PG native snapshot")
    for name in ("cells.tsv", "pins.tsv", "ports.tsv", "native.log"):
        if sha(folder / name) != result["artifacts"][name]:
            raise ValueError("Changed native snapshot: " + name)
    return dispatch, result, cells(folder / "cells.tsv")


def ports_geometry(path):
    import csv
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    # Clock boundary net names may change during real CTS. Port identity,
    # direction and physical box geometry may not.
    return sorted(tuple(row[k] for k in ("port", "direction", "signal_type", "layer",
                                        "xmin", "ymin", "xmax", "ymax", "status")) for row in rows)


def normalize(*, original_snapshot, candidate_snapshot, candidate_odb, output, image, timeout=1800):
    original_snapshot, candidate_snapshot = Path(original_snapshot), Path(candidate_snapshot)
    old_dispatch, _, before = load_snapshot(original_snapshot)
    new_dispatch, _, after = load_snapshot(candidate_snapshot)
    if old_dispatch["stage"] != "60_opt" or old_dispatch["die"] != new_dispatch["die"]:
        raise ValueError("Use matching-die frozen post-OPT input")
    if old_dispatch["image"] != image or new_dispatch["image"] != image:
        raise ValueError("Independent inspection/legalization runtime differs")
    if old_dispatch["libraries"] != new_dispatch["libraries"]:
        raise ValueError("Input/candidate technology changed")
    original_sdc = old_dispatch["input_hashes"]["60_opt/" + old_dispatch["die"] + "/design.sdc"]
    if new_dispatch.get("clock_sdc_sha256") != original_sdc:
        raise ValueError("Candidate clock classification must use the protected original SDC")
    if ports_geometry(original_snapshot / "ports.tsv") != ports_geometry(candidate_snapshot / "ports.tsv"):
        raise ValueError("Original port geometry or identity changed")
    restore, removed_clock = restoration(before, after)
    odb = Path(candidate_odb).resolve(strict=True)
    expected_odb = new_dispatch.get("input_odb_sha256")
    if expected_odb is None:
        expected_odb = new_dispatch["input_hashes"][new_dispatch["stage"] + "/" + new_dispatch["die"] + "/design.odb"]
    if sha(odb) != expected_odb:
        raise ValueError("Candidate ODB differs from independently inspected ODB")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Use a pinned independent runtime")
    output = Path(output).absolute()
    for protected in (odb.parent, original_snapshot.resolve(), candidate_snapshot.resolve(), NATIVE.resolve()):
        if path_within(output.resolve(), protected):
            raise ValueError("Legalization output overlaps protected evidence/input")
    output.mkdir(parents=True, exist_ok=False)
    control = output / "control"
    control.mkdir()
    (control / "restore.tsv").write_text(restore)
    result_dir = output / "native"
    result_dir.mkdir()
    name = "o3db-engineering-legalize-" + uuid.uuid4().hex[:16]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", "8", "--memory", "32g",
        "--pids-limit", "512", "--tmpfs", "/tmp:rw,size=1g", "--user", f"{os.getuid()}:{os.getgid()}"]
    for path, target, ro in ((odb, "/input/design.odb", True), (NATIVE, "/trusted", True),
                              (control, "/control", True), (result_dir, "/out", False)):
        command += bind(path, target, ro)
    for item in ("ENG_ODB=/input/design.odb", "ENG_RESTORE=/control/restore.tsv", "ENG_OUTPUT=/out", "HOME=/tmp"):
        command += ["--env", item]
    command += ["--entrypoint", "/usr/local/bin/openroad", image, "-no_init", "-exit", "/trusted/legalize.tcl"]
    write_new(output / "dispatch.json", {"kind": "engineering-protected-legalization-v1", "policy": POLICY,
        "started_at": now(), "image": image, "source_odb_sha256": expected_odb,
        "original_snapshot_sha256": sha(original_snapshot / "result.json"),
        "candidate_snapshot_sha256": sha(candidate_snapshot / "result.json"),
        "restore_sha256": sha(control / "restore.tsv"), "script_sha256": sha(NATIVE / "legalize.tcl"),
        "command": command, "candidate_final": False, "task_qualification": False})
    started = time.monotonic()
    code = None
    try:
        with (output / "native.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        code = process.returncode
        text = (output / "native.log").read_text()
        marker = re.findall(r"^ENGINEERING_PROTECTED_LEGALIZATION restored=(\d+) pg=(\d+)$", text, re.M)
        if code or len(marker) != 1 or re.search(r"^\[(?:WARNING|ERROR)\b", text, re.M):
            raise ValueError("Protected legalization failed or emitted unaudited warnings")
        restored, pg = map(int, marker[0])
        if restored != sum(not c["clock_cell"] for c in before.values()):
            raise ValueError("Protected normalization count differs")
        if sha(odb) != expected_odb:
            raise ValueError("Read-only input changed")
        result = {"status": "succeeded", "restored_original_cells": restored,
                  "removed_original_clock_cells": removed_clock, "checked_pg_pins": pg,
                  "artifacts": {str(p.relative_to(output)): sha(p) for p in
                                (result_dir / "design.odb", result_dir / "design.def", output / "native.log")}}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            subprocess.run(["docker", "stop", "--time", "5", name], capture_output=True, timeout=20, check=False)
        result = {"status": "failed", "error": str(exc)}
    result.update(completed_at=now(), duration_seconds=time.monotonic() - started,
                  exit_code=code, candidate_final=False, task_qualification=False)
    write_new(output / "result.json", result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("original_snapshot", "candidate_snapshot", "candidate_odb", "output", "image"):
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    args = parser.parse_args(argv)
    result = normalize(**vars(args))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
