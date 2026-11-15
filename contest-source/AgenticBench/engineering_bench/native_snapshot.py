"""Bounded, read-only native ODB/Liberty/clock inventory in the pinned image.

Snapshots are diagnostic evidence, not successful task qualifications. No host
OpenROAD execution, source import, credentials, network or mutable reference.
"""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .protocol import sha, write_new, now, digest, path_within

NATIVE = Path(__file__).with_name("native")


def platform_libs(platform, die):
    directory = Path(platform) / ("lib_" + die)
    groups = {}
    for path in sorted(list(directory.glob("*.lib")) + list(directory.glob("*.lib.gz"))):
        key = path.name.replace(" copy.lib", ".lib")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            cells = len(re.findall(r"\bcell\s*\(", stream.read()))
        groups.setdefault(key, []).append((cells, " copy.lib" not in path.name, path))
    result = [max(values)[2] for key, values in sorted(groups.items())]
    if not result:
        raise ValueError("Missing platform Liberty files")
    return result


def cells(path):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            name = row["name"]
            if not name or name in result or None in row:
                raise ValueError("Malformed/repeated native cell")
            for key in ("x_dbu", "y_dbu", "clock_cell", "cts_owned", "buffer", "inverter"):
                row[key] = int(row[key])
            for key in ("clock_cell", "cts_owned", "buffer", "inverter"):
                if row[key] not in (0, 1):
                    raise ValueError("Invalid native cell classification")
            if row["clock_cell"] and not (row["buffer"] or row["inverter"]):
                raise ValueError("Clock role without Liberty proof")
            result[name] = row
    if not result:
        raise ValueError("Empty native inventory")
    return result


def bind(path, target, readonly=True):
    path = Path(path).resolve(strict=True)
    if any(c in str(path) for c in (",", "\n", "\x00")):
        raise ValueError("Unsafe Docker path")
    return ["--mount", f"type=bind,src={path},dst={target},readonly={'true' if readonly else 'false'}"]


def snapshot(*, run, platform, image, stage, die, output, timeout=300, clock_sdc=None, odb_path=None):
    run, platform, output = Path(run).resolve(strict=True), Path(platform).resolve(strict=True), Path(output).absolute()
    if stage not in {"60_opt", "70_cts"} or die not in {"bottom", "top"}:
        raise ValueError("Expected maintained OPT/CTS input and a die")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Native image must be pinned")
    for protected in (run, platform, NATIVE.resolve()):
        if path_within(output.resolve(), protected):
            raise ValueError("Snapshot output must not be inside a protected input")
    odb, sdc = run / stage / die / "design.odb", run / stage / die / "design.sdc"
    if odb_path is not None:
        odb = Path(odb_path).resolve(strict=True)
        if path_within(output.resolve(), odb.parent):
            raise ValueError("Snapshot output overlaps external ODB input")
    if clock_sdc is not None:
        sdc = Path(clock_sdc).resolve(strict=True)
    sdc.resolve(strict=True).relative_to(run)
    if odb_path is None:
        odb.resolve(strict=True).relative_to(run)
    libraries = platform_libs(platform, die)
    odb_key = str(odb.relative_to(run)) if odb_path is None else "external_odb"
    inputs = {odb_key: sha(odb), str(sdc.relative_to(run)): sha(sdc)}
    libs = {str(p.relative_to(platform)): sha(p) for p in libraries}
    output.mkdir(parents=True, exist_ok=False)
    container = "o3db-engineering-snapshot-" + uuid.uuid4().hex[:16]
    command = ["docker", "run", "--rm", "--name", container, "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--cpus", "2", "--memory", "16g", "--pids-limit", "512", "--tmpfs", "/tmp:rw,size=1g",
        "--user", f"{os.getuid()}:{os.getgid()}"]
    for path, target, readonly in ((run, "/input", True), (platform, "/platform", True),
                                   (NATIVE, "/trusted", True), (output, "/out", False)):
        command += bind(path, target, readonly)
    if odb_path is not None:
        command += bind(odb, "/candidate/design.odb")
    env = {"ENG_ODB": "/candidate/design.odb" if odb_path is not None else f"/input/{stage}/{die}/design.odb", "ENG_SDC": "/input/" + str(sdc.relative_to(run)),
           "ENG_LIBS": "\n".join("/platform/" + str(p.relative_to(platform)) for p in libraries),
           "ENG_OUTPUT": "/out", "HOME": "/tmp"}
    for key, value in env.items():
        command += ["--env", key + "=" + value]
    command += ["--entrypoint", "/usr/local/bin/openroad", image, "-no_init", "-exit", "/trusted/inspect.tcl"]
    write_new(output / "dispatch.json", {"kind": "engineering-native-snapshot-attempt-v1", "started_at": now(),
        "image": image, "stage": stage, "die": die, "input_hashes": inputs, "libraries": libs,
        "clock_sdc_sha256": sha(sdc),
        "input_odb_sha256": sha(odb), "external_odb": odb_path is not None,
        "script_sha256": sha(NATIVE / "inspect.tcl"), "command": command,
        "candidate_final": False, "task_qualification": False})
    started = time.monotonic()
    error, code = None, None
    try:
        with (output / "native.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        code = process.returncode
        text = (output / "native.log").read_text()
        marker = re.findall(r"^ENGINEERING_NATIVE_SNAPSHOT cells=(\d+) clocks=(\d+) pg=(\d+) floating_pg=(\d+)$", text, re.M)
        if code or len(marker) != 1 or re.search(r"^\[(?:WARNING|ERROR)\b", text, re.M):
            raise ValueError("Native snapshot failed or has unaudited warnings; inspect native.log")
        table = cells(output / "cells.tsv")
        count, clocks, pg, floating_pg = map(int, marker[0])
        if count != len(table) or clocks < 1:
            raise ValueError("Native inventory completeness mismatch")
        stats = {"cells": count, "clock_cells": sum(r["clock_cell"] for r in table.values()),
                 "cts_owned": sum(r["cts_owned"] for r in table.values()),
                 "clock_count": clocks, "pg_pins": pg, "floating_pg": floating_pg}
        if inputs != {odb_key: sha(odb), str(sdc.relative_to(run)): sha(sdc)}:
            raise ValueError("Snapshot inputs changed while reading")
        result = {"status": "succeeded", "statistics": stats,
                  "artifacts": {n: sha(output / n) for n in ("cells.tsv", "pins.tsv", "ports.tsv", "native.log")}}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            subprocess.run(["docker", "stop", "--time", "5", container], capture_output=True, timeout=20, check=False)
        result = {"status": "failed", "error": str(exc)}
    result.update(completed_at=now(), duration_seconds=time.monotonic() - started,
                  exit_code=code, candidate_final=False, task_qualification=False)
    write_new(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "platform", "image", "stage", "die", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--clock-sdc", help="Protected frozen input SDC for candidate clock classification")
    parser.add_argument("--odb-path", help="Explicit candidate ODB; never represented as a frozen checkpoint")
    args = parser.parse_args(argv)
    result = snapshot(**vars(args))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
