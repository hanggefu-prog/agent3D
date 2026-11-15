"""Trusted post-CTS continuous-passive power and HotSpot endpoint (no routing).

The original CTS STA script and inputs are immutable. Only report destinations
are redirected before appending power queries. This is not routed/signoff power.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

ENDPOINT = "post-cts-continuous-passive-hotspot-v1"
# Exact producer from release 0930956e. Its native calculation is unchanged by
# the host-side image-asset validation fix below; old evidence stays immutable.
REVIEWED_PRODUCER_SHA256 = "d7a91825e60a28b45a4d5106a295ca84bfc45708a4a4fc54a7b449f24530a351"
# Independently read from the pinned installed image, never from a candidate.
IMAGE_ASSETS = {
    "/opt/open3dbench-assets/hotspot/hotspot": (380520, "02acd591ff8ec4b2d72def9053dac327a98f9c51725582ff7db8f62aa4a50665"),
    "/opt/open3dbench-assets/hotspot/examples/thermal/test.config": (6031, "966f56f7f4110452837a25563c8114b4f03b3581e797f5ee4b2754a82f350b4e"),
    "/opt/open3dbench-assets/hotspot/examples/thermal/test.materials": (298, "d958a3263b1966599a4dddf6b1a943298ab0eb415ec4430214bca3b1a2047a82"),
}
OUTPUTS = {
    "CTS_ANNOTATION_REPORT": "parasitic_annotation.rpt",
    "CTS_BOUNDARY_CUTPOINTS_TSV": "boundary_cutpoints.tsv",
    "CTS_BOUNDARY_LOADS_TSV": "boundary_loads.tsv",
    "CTS_CHECK_SETUP_REPORT": "check_setup.rpt",
    "CTS_CLOCK_INTERFACE_TSV": "clock_hbt_interface.tsv",
    "CTS_HOLD_REPORT": "hold.rpt", "CTS_METRICS_TSV": "metrics.tsv",
    "CTS_SETUP_REPORT": "setup.rpt", "CTS_SINKS_BY_CLOCK_TSV": "clock_sinks_by_clock.tsv",
    "CTS_SINKS_TSV": "clock_sinks.tsv",
}
INPUTS = {"CTS_TECH_LEF", "CTS_DIE_LEFS", "CTS_LIBS", "OPEN3D_LIBERTY_SUPPRESSIONS",
          "CTS_PASSIVE_V", "CTS_PASSIVE_SDC", "CTS_PASSIVE_SPEF", "CTS_PASSIVE_PLAN"}
SCALARS = {"CTS_CLOCK_HBT_ID", "CTS_PRIMARY_CLOCK_NAME", "CTS_QUERY_ALL_BOUNDARIES", "CTS_EXPECTED_HBTS"}
POWER_TCL = r'''
if {[llength [get_cells -quiet HBT_TIMING_*]] != 0} { error "Passive HBT power cannot include adapter cells" }
set_power_activity -input -activity 0.1
report_power -format json -digits 12 > $::env(WORKSPACE_POWER_JSON)
report_power -instances [get_cells -hier *] -digits 12 > $::env(WORKSPACE_INSTANCE_POWER)
puts "OPEN3D_WORKSPACE_CTS_POWER_DONE"
'''


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def replay_environment(text, body):
    require(text.endswith(body), "CTS STA body differs from trusted implementation")
    prefix = text[:-len(body)]
    env, position = {}, 0
    for match in re.finditer(r"set ::env\(([A-Z0-9_]+)\) \{([^{}]*)\}\n", prefix):
        require(not prefix[position:match.start()].strip(), "Unexpected executable Tcl prefix")
        key, value = match.groups()
        require(key not in env, "Duplicate STA environment assignment")
        env[key] = value
        position = match.end()
    require(not prefix[position:].strip(), "Unexpected trailing Tcl prefix")
    require(set(env) == set(OUTPUTS) | INPUTS | SCALARS, "Unexpected CTS STA environment contract")
    require(env["CTS_QUERY_ALL_BOUNDARIES"] in {"0", "1"}, "Invalid boundary query flag")
    require(int(env["CTS_EXPECTED_HBTS"]) > 0, "Missing HBT inventory")
    return env


def normalize_def(source, destination):
    """Preserve every component and its placement for the frozen DEF reader."""
    text = Path(source).read_text()
    units = re.search(r"^UNITS DISTANCE MICRONS\s+(\d+)\s*;", text, re.M)
    area = re.search(r"^DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;", text, re.M)
    section = re.search(r"^COMPONENTS\s+(\d+)\s*;\s*\n(.*?)^END COMPONENTS\s*$", text, re.M | re.S)
    require(units and area and section, "Missing DEF units/outline/components")
    statements = [s.strip() for s in section.group(2).split(";") if s.strip()]
    require(len(statements) == int(section.group(1)), "DEF declared component count differs")
    rows = {}
    for statement in statements:
        identity = re.search(r"^-\s+(\S+)\s+(\S+)", statement)
        place = re.search(r"\+\s*(PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)", statement)
        require(identity and place, "Unplaced/malformed DEF component")
        name, master = identity.groups()
        require(name not in rows, "Duplicate DEF component")
        rows[name] = (master, *place.groups())
    box = tuple(map(int, area.groups()))
    require(int(units.group(1)) > 0 and box[2] > box[0] and box[3] > box[1], "Invalid DEF geometry")
    lines = ['VERSION 5.8 ;', 'DIVIDERCHAR "/" ;', 'BUSBITCHARS "[]" ;',
             'DESIGN thermal_snapshot ;', units.group(0), area.group(0),
             'COMPONENTS {} ;'.format(len(rows))]
    for name, (master, status, x, y, orient) in sorted(rows.items()):
        lines.append('- {} {} + {} ( {} {} ) {} ;'.format(name, master, status, x, y, orient))
    lines.extend(['END COMPONENTS', 'END DESIGN', ''])
    Path(destination).write_text('\n'.join(lines))
    return {"source": record(source), "normalized": record(destination), "components": len(rows)}


def validate_power(instance_path, global_path, parser):
    for line in Path(instance_path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            float(fields[0])
        except ValueError:
            require(not any(n in line for n in ("u_bottom/", "u_top/", "HBT_TIMING_")), "Malformed instance power row")
            continue
        require(len(fields) == 5, "Malformed numeric instance power row")
        values = list(map(float, fields[:4]))
        require(all(math.isfinite(v) and v >= 0 for v in values), "Nonfinite/negative power")
        require(abs(sum(values[:3]) - values[3]) <= max(1e-12, values[3] * 1e-5), "Instance power does not reconcile")
        require(not fields[4].startswith("HBT_TIMING_"), "Unexpected active HBT adapter")
    powers, hbt = parser(Path(instance_path))
    require(hbt == 0 and set(powers) == {"bottom", "top"}, "Invalid passive-HBT die inventory")
    canonical = {}
    for die, rows in powers.items():
        canonical[die] = {}
        for name, value in rows.items():
            name = name.replace(r"\/", "/")
            require(name not in canonical[die], "Escaped instance name collision")
            canonical[die][name] = value
        require(bool(rows), "Empty die power inventory")
    total = json.loads(Path(global_path).read_text())["Total"]
    values = {key: float(total[key]) for key in ("internal", "switching", "leakage", "total")}
    require(all(math.isfinite(v) and v >= 0 for v in values.values()) and values["total"] > 0, "Invalid global power")
    require(abs(sum(values[k] for k in ("internal", "switching", "leakage")) - values["total"]) <= max(1e-12, values["total"] * 1e-5), "Global power components differ")
    difference = abs(sum(sum(v.values()) for v in canonical.values()) - values["total"])
    require(difference <= max(1e-12, values["total"] * 1e-4), "Instance/global power differs by more than 100 ppm")
    return canonical, values, difference


def validate_tiles(path):
    expected = {"layer_{}_{}_{}_{}".format(layer, die, x, y)
                for layer, die in ((0, "top"), (2, "bottom")) for x in range(10) for y in range(10)}
    observed = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith(("layer_0_top_", "layer_2_bottom_")):
            require(len(fields) == 2 and math.isfinite(float(fields[1])) and float(fields[1]) >= 318.15, "Invalid thermal tile")
            observed.append(fields[0])
    require(set(observed) == expected and len(observed) == 200, "HotSpot must publish exactly 200 active tiles")


def validate_snapshot(directory, checkpoint):
    """Artifact-only consumer: independently recompute temperature and power."""
    directory, checkpoint = Path(directory).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    manifest = json.loads((directory / "snapshot_manifest.json").read_text())
    require(manifest.get("schema_version") == 1 and manifest.get("status") == "success"
            and manifest.get("endpoint") == ENDPOINT and manifest.get("source_stage") == "cts",
            "Invalid thermal snapshot identity")
    require(manifest.get("implementation_sha256") in {sha(__file__), REVIEWED_PRODUCER_SHA256}, "Thermal implementation differs")
    require(manifest.get("physical_thermal_sha256") == "250976bed5bc7560927019ef4709a01c79346f37b5668a889bacc1391a9fa4e8", "Physical thermal mapping differs")
    require(manifest.get("source_checkpoint") == record(checkpoint), "Thermal checkpoint changed")
    artifacts = manifest.get("artifacts", {})
    required = {"metrics.json", "power.ptrace", "temperature.steady", "sta/power.json", "sta/instance_power.rpt",
                "sta/metrics.tsv", "sta/sta.log", "sta/replay.tcl", "stack.lcf", "top.flp", "bottom.flp",
                "hotspot.config", "materials.txt", "hotspot.log", "normalized_defs/top.def", "normalized_defs/bottom.def"}
    require(required <= set(artifacts), "Incomplete thermal artifacts")
    for name, row in artifacts.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe thermal artifact")
        path = (directory / relative).resolve(strict=True)
        require(directory in path.parents and record(path) == row, "Thermal artifact changed")
    require(bool(manifest.get("inputs")), "Missing thermal input provenance")
    require(set(IMAGE_ASSETS) <= set(manifest["inputs"]), "Missing installed HotSpot asset evidence")
    for name, row in manifest["inputs"].items():
        if name in IMAGE_ASSETS:
            size, expected = IMAGE_ASSETS[name]
            require(row == {"path": name, "bytes": size, "sha256": expected}, "Installed HotSpot asset differs")
            # On the host these paths do not exist. In the verifier container,
            # additionally compare actual bytes; never accept a different path.
            if Path(name).exists():
                require(record(name) == row, "Installed HotSpot asset changed")
        else:
            require(record(name) == row, "Thermal input changed")
    for local, installed in (("hotspot.config", "test.config"), ("materials.txt", "test.materials")):
        require(sha(directory / local) == IMAGE_ASSETS["/opt/open3dbench-assets/hotspot/examples/thermal/" + installed][1], "HotSpot copied configuration differs")
    metrics = json.loads((directory / "metrics.json").read_text())
    require(all(metrics.get(k) == v for k, v in {"stage": "thermal_snapshot", "status": "success", "endpoint": ENDPOINT,
            "source_stage": "cts", "algorithm": "hotspot-steady-grid", "grid_size": 10, "ambient_k": 318.15,
            "warning_count": 0, "signoff": False, "routed": False}.items()), "Thermal metric contract differs")
    provenance = metrics.get("power", {}).get("provenance")
    require(provenance == {"model": "continuous-passive-hbt-placement-estimated", "checkpoint": str(checkpoint), "threads": 4}, "Power provenance differs")
    require(metrics.get("stack") == {"order_from_layer_zero": ["top-silicon", "tim", "bottom-silicon", "tim"],
            "silicon_thickness_um": 150.0, "tim_thickness_um": 20.0,
            "model_status": "provisional-shared-baseline"}, "Thermal stack contract differs")
    require("OPEN3D_WORKSPACE_CTS_POWER_DONE" in (directory / "sta/sta.log").read_text(), "Missing native power completion")
    require(not re.search(r"\b(warning|error)\b", (directory / "hotspot.log").read_text(), re.I), "HotSpot warning/error")
    def timing_values(path):
        values = {}
        for line in Path(path).read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                require(fields[0] not in values, "Duplicate STA metric")
                values[fields[0]] = float(fields[1])
        require(bool(values) and all(math.isfinite(v) for v in values.values()), "Missing/nonfinite STA metrics")
        return values
    before = timing_values(checkpoint.parent / "assembly/passive/metrics.tsv")
    after = timing_values(directory / "sta/metrics.tsv")
    require(set(before) == set(after) and all(math.isclose(before[k], after[k], rel_tol=1e-10, abs_tol=1e-9) for k in before), "Native STA replay differs from CTS")
    validate_tiles(directory / "temperature.steady")
    dies = {"top": [], "bottom": []}
    for line in (directory / "temperature.steady").read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("layer_0_top_"):
            dies["top"].append(float(fields[1]))
        elif fields and fields[0].startswith("layer_2_bottom_"):
            dies["bottom"].append(float(fields[1]))
    temperature = metrics["temperature"]
    for die, values in dies.items():
        require(temperature[die]["tile_count"] == 100 and math.isclose(temperature[die]["average_k"], sum(values) / 100, abs_tol=1e-8, rel_tol=0), "Thermal tile mean differs")
    all_values = dies["top"] + dies["bottom"]
    average, maximum = sum(all_values) / 200, max(all_values) - 273.15
    require(math.isclose(temperature["global_average_k"], average, abs_tol=1e-8, rel_tol=0)
            and math.isclose(temperature["global_maximum_c"], maximum, abs_tol=1e-8, rel_tol=0), "Global temperature differs")
    lines = (directory / "power.ptrace").read_text().splitlines()
    require(len(lines) == 2, "Malformed power trace")
    names = ["{}_{}_{}".format(d, x, y) for d in ("top", "bottom") for x in range(10) for y in range(10)]
    values = list(map(float, lines[1].split()))
    require(lines[0].split() == names and len(values) == 200 and all(math.isfinite(v) and v >= 0 for v in values), "Invalid power trace")
    instance_values = []
    for line in (directory / "sta/instance_power.rpt").read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            float(fields[0])
        except ValueError:
            continue
        require(len(fields) == 5, "Malformed instance power")
        value = float(fields[3]); require(math.isfinite(value) and value >= 0, "Invalid instance power")
        instance_values.append(value)
    total = sum(instance_values)
    require(total > 0 and abs(sum(values) - total) <= max(1e-12, total * 1e-9), "Native instance/tile power differs")
    require(abs(metrics["power"]["total_w"] - total) <= max(1e-12, total * 1e-9), "Published power differs")
    global_total = float(json.loads((directory / "sta/power.json").read_text())["Total"]["total"])
    require(math.isfinite(global_total) and global_total > 0 and abs(global_total - total) <= max(1e-12, global_total * 1e-4), "Native global/instance power differs")
    return {"global_average_temperature_k": average, "global_average_temperature_c": average - 273.15,
            "bottom_average_temperature_k": sum(dies["bottom"]) / 100,
            "top_average_temperature_k": sum(dies["top"]) / 100,
            "global_maximum_temperature_c": maximum, "total_power_w": total, "stack": metrics["stack"]}


def run(checkpoint, platform, hotspot, output):
    from open3dbench.backend import thermal as frozen
    from open3dbench.backend.three_d_cts import PASSIVE_CTS_STA_TCL
    from open3dbench.backend.three_d_opt import _environment_script, _run_openroad, _read_key_values
    started = time.monotonic()
    checkpoint, platform, hotspot, output = (Path(p).resolve() for p in (checkpoint, platform, hotspot, output))
    require(not output.exists(), "Thermal output must be new; inspect prior attempts")
    require(checkpoint.parent not in output.parents, "Thermal output cannot modify CTS")
    cp = frozen.validate_checkpoint(checkpoint, require_files=True)
    require(cp.get("stage") == "cts" and cp.get("ready_for_route") is True and cp.get("pdk") == "ng45", "Expected ready ng45 CTS checkpoint")
    cts = checkpoint.parent
    env = replay_environment((cts / "assembly/passive/validate_sta.tcl").read_text(), PASSIVE_CTS_STA_TCL)
    inputs = {str(checkpoint): record(checkpoint)}
    for key in INPUTS:
        for item in env[key].splitlines():
            path = Path(item).resolve(strict=True)
            require(output != path and output not in path.parents, "Input overlaps writable output")
            inputs[str(path)] = record(path)
    for key, name in (("CTS_PASSIVE_V", "design.v"), ("CTS_PASSIVE_SDC", "design.sdc"),
                      ("CTS_PASSIVE_SPEF", "design.spef"), ("CTS_PASSIVE_PLAN", "plan.tsv")):
        require(sha(env[key]) == sha(cts / "assembly/passive" / name), "CTS selected iteration differs from final assembly")
    output.mkdir(parents=True)
    sta = output / "sta"; sta.mkdir()
    relocated = {**env, **{key: str(sta / name) for key, name in OUTPUTS.items()},
                 "WORKSPACE_POWER_JSON": str(sta / "power.json"), "WORKSPACE_INSTANCE_POWER": str(sta / "instance_power.rpt")}
    script = _environment_script(relocated, PASSIVE_CTS_STA_TCL + POWER_TCL)
    (sta / "replay.tcl").write_text(script)
    power_start = time.monotonic()
    audit = _run_openroad(script, {"OPEN3DBENCH_THREADS": "4"}, sta / "sta.log", "/usr/local/bin/openroad")
    power_seconds = time.monotonic() - power_start
    require("OPEN3D_WORKSPACE_CTS_POWER_DONE" in (sta / "sta.log").read_text(), "Missing native power completion")
    before = _read_key_values(cts / "assembly/passive/metrics.tsv")
    after = _read_key_values(sta / "metrics.tsv")
    require(set(before) == set(after) and all(math.isfinite(float(after[k])) and math.isclose(float(before[k]), float(after[k]), rel_tol=1e-10, abs_tol=1e-9) for k in before), "STA replay differs from CTS")
    powers, global_power, difference = validate_power(sta / "instance_power.rpt", sta / "power.json", frozen._read_instance_power)
    normalized = output / "normalized_defs"; normalized.mkdir()
    geometry, grids, normalization = {}, {}, {}
    for die in ("bottom", "top"):
        for path in sorted((platform / ("lef_" + die)).glob("*.lef")):
            inputs[str(path)] = record(path)
    sizes = frozen._lef_sizes(platform)
    for die in ("top", "bottom"):
        source = (cts / cp["dies"][die]["def"]).resolve(strict=True)
        inputs[str(source)] = record(source)
        target = normalized / (die + ".def")
        normalization[die] = normalize_def(source, target)
        _, _, parsed = frozen._read_def(target)
        require(len(parsed) == normalization[die]["components"], "Frozen DEF reader lost components")
        grids[die], geometry[die] = frozen._grid_power(die, powers[die], target, sizes, 10)
        frozen._write_floorplan(output / (die + ".flp"), die, geometry[die], 10)
    require(all(geometry["top"][k] == geometry["bottom"][k] for k in ("diearea_dbu", "dbu_per_micron")), "Thermal die geometry differs")
    names = ["{}_{}_{}".format(d, x, y) for d in ("top", "bottom") for x in range(10) for y in range(10)]
    values = [grids[d]["{}_{}_{}".format(d, x, y)] for d in ("top", "bottom") for x in range(10) for y in range(10)]
    (output / "power.ptrace").write_text(" ".join(names) + "\n" + " ".join("{:.12g}".format(v) for v in values) + "\n")
    frozen._write_stack(output / "stack.lcf")
    for name, path in (("hotspot.config", hotspot / "examples/thermal/test.config"), ("materials.txt", hotspot / "examples/thermal/test.materials")):
        inputs[str(path)] = record(path); shutil.copyfile(path, output / name)
    inputs[str(hotspot / "hotspot")] = record(hotspot / "hotspot")
    command = [str(hotspot / "hotspot"), "-c", "hotspot.config", "-p", "power.ptrace", "-grid_layer_file", "stack.lcf", "-materials_file", "materials.txt", "-model_type", "grid", "-detailed_3D", "on", "-ambient", "318.15", "-init_temp", "318.15", "-steady_file", "temperature.steady", "-grid_steady_file", "temperature.grid.steady"]
    hotspot_start = time.monotonic()
    with (output / "hotspot.log").open("x") as log:
        result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
    hotspot_seconds = time.monotonic() - hotspot_start
    require(result.returncode == 0 and not re.search(r"\b(warning|error)\b", (output / "hotspot.log").read_text(), re.I), "HotSpot failed or warned")
    validate_tiles(output / "temperature.steady")
    total = sum(sum(v.values()) for v in powers.values())
    error = abs(sum(values) - total)
    require(total > 0 and error <= max(1e-12, total * 1e-9), "Tile power is not conserved")
    require(all(sha(path) == row["sha256"] for path, row in inputs.items()), "Immutable thermal input changed")
    temperature = frozen._temperature_metrics(output / "temperature.steady", 318.15)
    temperature["global_average_k"] = sum(temperature[d]["average_k"] for d in ("bottom", "top")) / 2
    temperature["global_average_c"] = temperature["global_average_k"] - 273.15
    metrics = {"schema_version": 1, "stage": "thermal_snapshot", "status": "success", "endpoint": ENDPOINT,
        "source_stage": "cts", "algorithm": "hotspot-steady-grid", "grid_size": 10, "ambient_k": 318.15,
        "warning_count": 0, "signoff": False, "routed": False,
        "temperature": temperature,
        "power": {"total_w": total, "conservation_error_w": error, "global_report": global_power,
            "instance_global_difference_w": difference, "activity_model": "vectorless-primary-input-0.1-placement-estimated-rc",
            "provenance": {"model": "continuous-passive-hbt-placement-estimated", "checkpoint": str(checkpoint), "threads": 4}},
        "stack": {"order_from_layer_zero": ["top-silicon", "tim", "bottom-silicon", "tim"], "silicon_thickness_um": 150.0, "tim_thickness_um": 20.0, "model_status": frozen.THERMAL_STACK_MODEL_STATUS},
        "geometry": geometry, "normalization": normalization, "sta_warning_audit": audit,
        "runtime_seconds": time.monotonic() - started, "power_runtime_seconds": power_seconds, "hotspot_runtime_seconds": hotspot_seconds}
    write(output / "metrics.json", metrics)
    write(output / "snapshot_manifest.json", {"schema_version": 1, "status": "success", "endpoint": ENDPOINT,
        "source_stage": "cts", "source_checkpoint": record(checkpoint), "inputs": inputs,
        "artifacts": {str(p.relative_to(output)): record(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "implementation_sha256": sha(__file__), "physical_thermal_sha256": sha(frozen.__file__)})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "platform-dir", "hotspot-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.platform_dir, args.hotspot_root, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
