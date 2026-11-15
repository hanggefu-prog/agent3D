"""Trusted HBT-only evaluation over a fixed five-stage physical prefix.

The candidate supplies two scalar functions. Geometry, materialization and STA
remain owned by Open3DBench; the verifier reads native evidence independently.
"""
import csv
import hashlib
import math
import os
import json
import shutil
import re
from collections import defaultdict
from pathlib import Path

from .hbt_policy import score_timing, validate_policy
from .reference_latest import read_json, require, sha256, validate_artifacts

PREFIX = {"input": "00_input", "floorplan": "10_floorplan", "partition": "20_partition",
          "placement": "30_placement", "pdn": "40_pdn"}
STA = "55_post_placement_sta"


def validate_recipe(config, source):
    require(config.get("type") == "lol" and config.get("pdk") in {"ng45", "asap7"},
            "HBT policy requires a maintained LoL platform")
    require(config.get("threads") == 4, "HBT policy requires four physical threads")
    hbt = config.get("hbt", {})
    require(hbt.get("algorithm") == "track-aware-greedy", "HBT policy requires track-aware-greedy")
    require(set(hbt) <= {"algorithm", "policy"}, "HBT policy cannot change grid, ideals or search")
    from .post_placement_timing import validate_recipe as partition_recipe
    partition_recipe(config, source, expected_decoder="global-2way")


def prefix_identity(run):
    from open3dbench.stage_manifest import validate_stage_manifest
    from open3dbench.backend.contracts import validate_checkpoint
    from .placement_baseline import validate_tp_gnn_partition
    root = Path(run).resolve(strict=True)
    identities = {}
    design = None
    previous = "none"
    for stage, directory in PREFIX.items():
        path = root / directory / "stage_manifest.json"
        value = validate_stage_manifest(path, expected_stage=stage, verify_artifacts=True)
        identity = (value.get("design"), value.get("integration"), value.get("pdk"))
        require(identity[0] and identity[1] == "lol" and identity[2] in {"ng45", "asap7"},
                "HBT input has an unsupported design identity")
        if design is None:
            design = identity
        require(identity == design, "HBT prefix mixes design identities")
        require(value["signature"].get("upstream") == previous,
                "HBT prefix has a broken upstream manifest chain: " + stage)
        identities[stage] = sha256(path)
        previous = identities[stage]
    validate_tp_gnn_partition(root, expected_method="tp-gnn-flat", require_bound_report=True,
                              expected_decoder="global-2way")
    checkpoint = validate_checkpoint(root / "40_pdn/checkpoint.json", require_files=True)
    require(checkpoint.get("stage") == "pdn", "HBT policy requires a physical PDN checkpoint")
    require(checkpoint.get("pdk") == design[2], "PDN checkpoint platform differs")
    return identities


def validate_input_identity(run, config):
    identity = prefix_identity(run)
    manifest = read_json(Path(run) / "40_pdn/stage_manifest.json")
    require((config.get("design"), config.get("type"), config.get("pdk"))
            == (manifest["design"], manifest["integration"], manifest["pdk"]),
            "HBT evaluation configuration differs from its frozen design/platform")
    return identity



def copy_prefix(source, destination):
    """Copy an authenticated fixed input without rebasing or resealing its metadata."""
    source, destination = Path(source).resolve(strict=True), Path(destination)
    expected = prefix_identity(source)
    require(not destination.exists(), "HBT evaluation output must be new")
    for directory in PREFIX.values():
        for path in (source / directory).rglob("*"):
            if path.is_symlink():
                target = path.resolve(strict=True)
                require(not Path(os.readlink(path)).is_absolute()
                        and any(target == source / name or source / name in target.parents for name in PREFIX.values()),
                        "HBT prefix contains an external or downstream alias")
    destination.mkdir(parents=True)
    for directory in PREFIX.values():
        shutil.copytree(source / directory, destination / directory, symlinks=True)
    require(prefix_identity(destination) == expected, "HBT input bytes changed during copy")
    return {"mode": "fixed-checkpoint-input", "stage_manifest_sha256": expected,
            "source": str(source), "executed_upstream_stages": []}


def implementation_identity(spec, source):
    root = Path(source).resolve(strict=True)
    # Include the complete physical Python implementation and platform contract,
    # not just the editable policy or the parent Git revision.
    paths = set()
    for folder in ("open3dbench", "Place-3D/src/hbt"):
        paths.update((root / folder).rglob("*.py"))
    for pdk in ("nangate45_3D", "asap7_3D"):
        folder = root / "OpenROAD-3D/flow/platforms" / pdk
        paths.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".json", ".tcl", ".lef", ".lib", ".gz"})
    result = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result["harness/" + path.name] = sha256(path)
    for name in ("task.yaml", "program_adapter.py", "initial_program.py"):
        result["task/" + name] = sha256(spec.root / name)
    executable = Path(os.environ.get("OPENROAD_EXE", "/usr/local/bin/openroad"))
    require(executable.is_file(), "HBT qualification requires the installed OpenROAD runtime")
    image = os.environ.get("OPEN3DBENCH_IMAGE_ID", "")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", image) is not None,
            "HBT qualification requires a pinned runtime image")
    result["runtime/openroad"] = sha256(executable.resolve(strict=True))
    result["runtime/image"] = image
    return result


def _rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _integer(value):
    require(not isinstance(value, bool), "Boolean geometry is invalid")
    result = int(value)
    require(str(result) == str(value), "HBT geometry must use integer DBU")
    return result


def validate_assignment(stage):
    """Check complete assignment, lattice/body legality, escape tracks and HPWL."""
    stage = Path(stage)
    problem = read_json(stage / "problem.json")
    require(problem.get("schema_version") == 1 and problem.get("units") == "dbu",
            "HBT problem units or schema differ")
    grid = problem["site_grid"]
    die = grid["die"]
    px, py, width, height = (_integer(grid[k]) for k in ("pitch_x", "pitch_y", "hbt_width", "hbt_height"))
    require(min(px, py, width, height) > 0 and px >= width + grid.get("spacing_x", 0)
            and py >= height + grid.get("spacing_y", 0), "Invalid HBT lattice")
    nets = {n["net_name"]: n for n in problem["nets"]}
    require(nets and len(nets) == len(problem["nets"]), "Duplicate or missing HBT networks")
    records = _rows(stage / "assignment/hbt.csv")
    require(len(records) == len(nets) and {r["net_name"] for r in records} == set(nets),
            "HBT assignment must cover each required network once")
    occupied, x_tracks, y_tracks = set(), defaultdict(list), defaultdict(list)
    coordinates, displacement = {}, []
    ideal_total = legal_total = 0

    def hpwl(pins, point):
        require(pins and all(len(p) == 2 for p in pins), "Missing HBT endpoint geometry")
        points = [tuple(_integer(v) for v in p) for p in pins] + [point]
        return sum(max(p[i] for p in points) - min(p[i] for p in points) for i in (0, 1))

    for row in records:
        x, y, ix, iy = (_integer(row[k]) for k in ("legal_x", "legal_y", "desired_x", "desired_y"))
        require((x, y) not in occupied, "HBT site is occupied twice")
        require(x >= max(die["x_min"], grid["origin_x"]) and y >= max(die["y_min"], grid["origin_y"])
                and x + width <= die["x_max"] and y + height <= die["y_max"]
                and (x-grid["origin_x"]) % px == 0 and (y-grid["origin_y"]) % py == 0,
                "HBT assignment is outside the legal grid")
        for box in problem.get("blockages", []):
            require(x+width <= box["x_min"] or box["x_max"] <= x
                    or y+height <= box["y_min"] or box["y_max"] <= y, "HBT intersects a blockage")
        occupied.add((x, y)); x_tracks[x].append(y); y_tracks[y].append(x)
        coordinates[row["net_name"]] = (x, y)
        net = nets[row["net_name"]]
        bottom, top = net["bottom_pins"], net["top_pins"]
        # The reference ideal is the center of the two bounding-box optimal interval.
        ideal = []
        for axis in (0, 1):
            lo = max(min(p[axis] for p in bottom), min(p[axis] for p in top))
            hi = min(max(p[axis] for p in bottom), max(p[axis] for p in top))
            ideal.append((lo + hi) // 2)
        require((ix, iy) == tuple(ideal), "HBT policy changed the fixed ideal")
        ih = hpwl(bottom, (ix, iy)) + hpwl(top, (ix, iy))
        lh = hpwl(bottom, (x, y)) + hpwl(top, (x, y))
        d = abs(x-ix) + abs(y-iy)
        expected = {"dx": x-ix, "dy": y-iy, "manhattan_displacement": d,
                    "objective_cost": lh, "ideal_combined_hpwl": ih, "legal_combined_hpwl": lh,
                    "combined_hpwl_increase": lh-ih}
        require(all(_integer(row[k]) == v for k, v in expected.items()), "HBT geometry metrics disagree")
        ideal_total += ih; legal_total += lh; displacement.append(d)
    for tracks, gap in ((x_tracks, 3*py), (y_tracks, 3*px)):
        for values in tracks.values():
            values = sorted(values)
            require(all(b-a >= gap for a, b in zip(values, values[1:])), "HBT escape tracks conflict")
    physical = _rows(stage / "hbt.csv")
    require(len(physical) == len(nets) and {r["net_name"] for r in physical} == set(nets),
            "Materialized HBT coverage differs")
    require(all((_integer(r["x_dbu"]), _integer(r["y_dbu"])) == coordinates[r["net_name"]]
                for r in physical), "Materialized HBT coordinates differ")
    result = {"assigned_count": len(nets), "unassigned_count": 0,
              "ideal_combined_hpwl_dbu": ideal_total, "legal_combined_hpwl_dbu": legal_total,
              "combined_hpwl_increase_dbu": legal_total-ideal_total,
              "total_displacement_dbu": sum(displacement), "mean_displacement_dbu": sum(displacement)/len(nets),
              "max_displacement_dbu": max(displacement), "unique_x_tracks": len(x_tracks), "unique_y_tracks": len(y_tracks),
              "shared_x_track_count": len(nets)-len(x_tracks), "shared_y_track_count": len(nets)-len(y_tracks)}
    metrics = read_json(stage / "assignment/metrics.json")
    require(all(metrics.get(k) == v for k, v in result.items()), "HBT native metrics disagree with geometry")
    policy = read_json(stage / "policy.json")
    validate_policy(policy["source"])
    require(policy == read_json(stage / "assignment/policy.json"), "HBT policy copy differs")
    evidence = metrics.get("policy", {})
    manifest = read_json(stage / "assignment/manifest.json")
    require(manifest.get("status") == "success" and manifest.get("algorithm") == "track-aware-greedy"
            and manifest.get("input_sha256") == sha256(stage / "problem.json")
            and manifest.get("policy") == evidence, "HBT assignment manifest differs")
    require(metrics.get("ideal_method") == "optimal-region" and metrics.get("legalizer") == "track-aware-routing-greedy",
            "HBT assignment method differs")
    require(evidence.get("source_sha256") == hashlib.sha256(policy["source"].encode()).hexdigest()
            and evidence.get("payload_sha256") == sha256(stage / "policy.json")
            and evidence.get("nearest_per_seed") == 8
            and evidence.get("calls", {}).get("net_priority") == len(nets)
            and type(evidence.get("calls", {}).get("site_cost")) is int
            and evidence["calls"]["site_cost"] >= len(nets), "Missing real HBT hook evidence")
    runtime = metrics.get("runtime_seconds")
    require(not isinstance(runtime, bool) and isinstance(runtime, (int, float))
            and math.isfinite(runtime) and runtime >= 0, "Invalid HBT runtime")
    result["hbt_assignment_runtime_seconds"] = runtime
    return result


def native_timing_values(path):
    """Read required numeric fields while preserving native textual diagnostics."""
    fields = {}
    for line in Path(path).read_text().splitlines():
        key, value = line.split("\t")
        require(key not in fields, "Duplicate native STA metric")
        fields[key] = value
    result = {}
    for key in ("wns_setup", "tns_setup", "clock_count"):
        require(key in fields, "Missing native STA metric: " + key)
        value = float(fields[key])
        require(math.isfinite(value), "Non-finite native STA metric: " + key)
        result[key] = value
    return result


def validate_result(run, *, frozen_input=None):
    from .post_placement_timing import validate_physical_snapshot
    root = Path(run).resolve(strict=True)
    identity = prefix_identity(root)
    if frozen_input is not None:
        require(identity == prefix_identity(frozen_input), "HBT candidate changed its frozen upstream prefix")
    manifest = read_json(root / "50_hbt/stage_manifest.json")
    upstream = read_json(root / "40_pdn/stage_manifest.json")
    require(all(manifest.get(k) == upstream.get(k) for k in ("design", "integration", "pdk"))
            and manifest.get("signature", {}).get("upstream") == identity["pdn"],
            "HBT result is not bound to its frozen PDN input")
    timing = validate_physical_snapshot(root)
    assignment = validate_assignment(root / "50_hbt")
    snapshot = read_json(root / STA / "timing.json")
    records = timing["artifact_records"]
    require(records.get("native_metrics", {}).get("path") == "passive/metrics.tsv",
            "HBT STA requires hash-bound native metrics")
    validate_artifacts(root / STA, records)
    raw = native_timing_values(root / STA / "passive/metrics.tsv")
    for key in ("wns_setup", "tns_setup", "clock_count"):
        value = raw.get(key)
        require(value is not None and math.isfinite(value) and timing["metrics"].get(key) == value
                and snapshot.get("metrics", {}).get(key) == value, "Native and summarized STA metrics disagree")
    require(raw["clock_count"] > 0, "HBT STA has no active clock")
    require(timing["hbt_counts"]["total"] == assignment["assigned_count"], "STA HBT coverage differs")
    return timing, assignment, identity


def timing_metrics(run, reference, *, frozen_input=None):
    row, assignment, prefix = validate_result(run, frozen_input=frozen_input)
    if reference is not None:
        other, _, other_prefix = validate_result(reference, frozen_input=frozen_input)
        require(prefix == other_prefix and row["hbt_model"] == other["hbt_model"],
                "HBT candidate/reference physical inputs differ")
        baseline = other["metrics"]
    else:
        baseline = row["metrics"]
    result = score_timing(row["metrics"], baseline)
    result.update(assignment)
    result.update(opt_passes=0.0, sta_graph_complete=1.0,
                  timing_evaluation_boundary="post-hbt-no-opt",
                  sta_runtime_seconds=row["runtime_seconds"])
    return result
