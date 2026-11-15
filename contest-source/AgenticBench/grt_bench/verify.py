"""Read-only result consumer. Recompute objectives from native evidence."""
from pathlib import Path
import math
import re
from . import ENDPOINT, TIMING_ENDPOINT, THERMAL_STATISTIC
from .common import finite, read, sha, verify_records
from .endpoint import validate_grt


def verify_physical(plan, task, case):
    from open3dbench.stage_manifest import validate_stage_manifest
    run = Path(plan.get("physical_output", plan["output"])) / "run"
    imported = {'input', 'floorplan'} if 'frontend_inputs' in task else set()
    runner = None
    if 'incremental' in task:
        from open3dbench.config import load_config
        from open3dbench.runner import FlowRunner
        config = load_config(run.parent/'resolved.yaml')
        config['run_dir'] = str(run)
        runner = FlowRunner(Path(plan['source']), config, resume=True)
    if plan.get('incremental'):
        from .incremental import verify_prefix
        imported = set(verify_prefix(plan, task, case, run, config)['stages'])
    if 'frontend_inputs' in task:
        from .frozen_frontend import verify
        verify(task, case, run)
    for stage, directory in (("input","00_input"),("floorplan","10_floorplan"),("partition","20_partition"),
            ("placement","30_placement"),("pdn","40_pdn"),("hbt","50_hbt"),("opt","60_opt"),("cts","70_cts")):
        manifest = validate_stage_manifest(run / directory / "stage_manifest.json", expected_stage=stage, verify_artifacts=True)
        if manifest["design"] != case["design"] or manifest["pdk"] != case["pdk"]:
            raise ValueError("Physical design/platform mismatch")
        if runner is not None and stage not in imported and not runner._resume_is_valid(stage):
            raise ValueError('Fresh suffix signature/artifact differs: ' + stage)
    observed = {"io": read(run / "10_floorplan/io_metrics.json")["seed"],
        "partition": read(run / "20_partition/partition_manifest.json")["seed"],
        "placement": read(run / "30_placement/stages/02_global_place/stage_manifest.json")["initialization_seed"]}
    if any(observed[k] != plan["seed"][k] for k in observed):
        raise ValueError("Native seed evidence differs")
    if 'incremental' in task:
        placement = read(run/'30_placement/stages/02_global_place/params.json')
        if placement.get('deterministic_flag') != 1:
            raise ValueError('Placement must use deterministic accumulation')
    if read(run / "60_opt/metrics.json").get("iterations_run") != task["opt_passes"]:
        raise ValueError("OPT pass count differs")
    validate_grt(run.parent / "grt/global_route_checkpoint.json")


def verify_cell(plan_path, result_path):
    from .common import digest, write, evaluator_identity
    from .contract import validate
    plan = read(plan_path)
    from .algorithm_scope import verify_runtime
    verify_runtime(plan)
    task = read(plan["task"])
    if validate(task) != plan["task_digest"]:
        raise ValueError("Task changed")
    if task["evaluator_sha256"] != evaluator_identity():
        raise ValueError("Evaluator differs from frozen task")
    case = next(c for c in task["cases"] if c["id"] == plan["case"])
    if plan["seed"] not in task["seeds"] or plan["split"] != case["split"]:
        raise ValueError("Case/seed/split changed")
    output = Path(plan["output"])
    if plan.get("rescore"):
        from .rescore import validate_plan
        validate_plan(plan, task)
    verify_physical(plan, task, case)
    metrics = verify_analysis(output / "analysis", expected_endpoint=task["endpoint"])
    receipt = {k: plan[k] for k in ("candidate", "case", "seed", "task_digest", "split")}
    receipt.update(status="verified", endpoint=task["endpoint"], metrics=metrics,
        evidence={"analysis_manifest_sha256":sha(output / "analysis/manifest.json"), "plan_sha256":sha(plan_path)})
    if plan.get("rescore"):
        receipt["physical_replay"] = plan["rescore"]
    if plan.get('incremental'):
        receipt['incremental'] = read(plan['incremental'])
    if "independent_overflow" in task:  # opt-in; frozen tasks never declare it
        from .independent_overflow import verify_hook
        receipt["independent_overflow"] = verify_hook(task, plan, output)
    write(result_path, receipt)
    return receipt


def verify_analysis(output, *, expected_endpoint=ENDPOINT):
    if expected_endpoint not in {ENDPOINT, TIMING_ENDPOINT}:
        raise ValueError("Unsupported analysis endpoint")
    timing_only = expected_endpoint == TIMING_ENDPOINT
    from open3dbench.backend import thermal
    from open3dbench.backend.three_d_opt import _read_key_values
    from open3dbench.backend.three_d_route import _grt_metrics
    from scripts.workspace_cts_thermal import validate_power, validate_tiles
    from . import endpoint
    output = Path(output).resolve(strict=True)
    manifest = read(output / "manifest.json")
    if manifest.get("endpoint") != expected_endpoint or manifest.get("implementation_sha256") != sha(endpoint.__file__):
        raise ValueError("Untrusted endpoint implementation")
    required = {"metrics.json", "assembly/metrics.tsv", "assembly/power.json", "assembly/instance_power.rpt",
                "assembly/design.spef", "assembly/check_setup.rpt", "assembly/latch_proof.tsv",
                "thermal/temperature.steady", "thermal/power.ptrace"}
    if timing_only:
        required -= {"assembly/power.json", "assembly/instance_power.rpt",
                     "thermal/temperature.steady", "thermal/power.ptrace"}
    if not required <= set(manifest["artifacts"]):
        raise ValueError("Incomplete endpoint evidence")
    verify_records(output, manifest["artifacts"])
    from .latch_audit import audit as audit_latches
    audit_latches(output / "assembly/check_setup.rpt", output / "assembly/latch_proof.tsv")
    for path, expected in manifest["inputs"].items():
        if sha(path) != expected:
            raise ValueError("Endpoint input changed")
    checkpoint = Path(manifest["source_checkpoint"])
    if sha(checkpoint) != manifest["source_checkpoint_sha256"]:
        raise ValueError("GRT checkpoint changed")
    validate_grt(checkpoint)
    timing = _read_key_values(output / "assembly/metrics.tsv")
    if timing.get("sta_completeness_checks_passed") != 1 or timing.get("hbt_adapter_cell_count") != 0:
        raise ValueError("Incomplete timing graph")
    if not timing_only:
        powers, _, _ = validate_power(output / "assembly/instance_power.rpt", output / "assembly/power.json", thermal._read_instance_power)
        validate_tiles(output / "thermal/temperature.steady")
        temperatures = [float(line.split()[1]) for line in (output / "thermal/temperature.steady").read_text().splitlines()
                        if line.split() and line.split()[0].startswith(("layer_0_top_", "layer_2_bottom_"))]
        ptrace = (output / "thermal/power.ptrace").read_text().splitlines()
        expected_tiles = {"{}_{}_{}".format(d, x, y) for d in ("top", "bottom") for x in range(10) for y in range(10)}
        if len(ptrace) != 2 or len(ptrace[0].split()) != 200 or set(ptrace[0].split()) != expected_tiles or len(ptrace[1].split()) != 200:
            raise ValueError("Incomplete tile power")
        total = sum(sum(d.values()) for d in powers.values())
        if not math.isclose(sum(map(float, ptrace[1].split())), total, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Power not conserved")
        sizes = thermal._lef_sizes(Path(manifest["platform"]))
        expected_power = {}
        for die in ("top", "bottom"):
            grid, _ = thermal._grid_power(die, powers[die], output / "thermal" / (die + ".def"), sizes, 10)
            expected_power.update(grid)
        for name, value in zip(ptrace[0].split(), ptrace[1].split()):
            if not math.isclose(finite(float(value), "tile power", 0), expected_power[name], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Spatial tile power differs from physical geometry")
        if re.search(r"\b(warning|error)\b", (output / "thermal/hotspot.log").read_text(), re.I):
            raise ValueError("HotSpot emitted a warning or error")
    actual = {"wns_deficit_ns": max(0, -finite(timing.get("wns_setup"), "WNS")),
        "tns_deficit_ns": max(0, -finite(timing.get("tns_setup"), "TNS")),
        "overflow": sum(_grt_metrics(checkpoint.parent / d)["grt_total_overflow"] for d in ("bottom", "top"))}
    if not timing_only:
        actual.update(temperature_peak_k=max(temperatures), power_w=total)
    recorded = read(output / "metrics.json")
    if recorded.get("endpoint") != expected_endpoint or recorded.get("parasitics") != "global-route-estimated-rc" or recorded.get("signoff") is not False:
        raise ValueError("Incorrect endpoint provenance")
    if ((not timing_only and recorded.get("thermal_statistic") != THERMAL_STATISTIC)
            or set(recorded.get("metrics", {})) != set(actual)):

        raise ValueError("Incorrect peak-temperature metric contract")
    for key, value in actual.items():
        if not math.isclose(finite(recorded["metrics"].get(key), key, 0), value, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Objective differs from native evidence: " + key)
    return actual


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("plan", type=Path)
    p.add_argument("result", type=Path)
    a = p.parse_args()
    verify_cell(a.plan, a.result)
