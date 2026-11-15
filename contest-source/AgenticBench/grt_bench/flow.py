"""In-container single-cell execution; harness supplies immutable plan/config.

Fresh full flow is the conservative default. A frontend-prefix may be requested
only through the physical runner's existing strict validator; no hash rewriting.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time

from .common import read, sha, write, evaluator_identity
from .contract import validate
from .endpoint import run


def resolved_config(base, case, seed):
    config = deepcopy(base)
    if config["design"] != case["design"] or config["pdk"] != case["pdk"] or config["type"] != "lol":
        raise ValueError("Case identity differs")
    for domain, key in (("pins", "io"), ("partition", "partition"), ("placement", "placement"), ("routing", "routing")):
        config.setdefault(domain, {})["seed"] = seed[key]
    config["thermal"] = {"enabled": False, "grid_size": 10, "ambient_k": 318.15}
    return config


def bind_method_seeds(config, seed, output):
    """Native adapters read their method JSON, not the top-level seed labels."""
    if config["partition"]["algorithm"] not in {"tp-gnn-flat", "hmetis", "mtkahypar", "random", "random-balanced"}:
        raise ValueError("Partition seed adapter is not qualified for this algorithm")
    if config["placement"]["algorithm"] != "concurrent-dreamplace":
        raise ValueError("Placement seed adapter is not qualified for this algorithm")
    for domain, group, key in (("partition", "partition", "seed"),
                               ("placement", "concurrent_dreamplace", "initialization_seed")):
        source = Path(config[domain]["config"])
        method = read(source)
        if not isinstance(method.get(group), dict):
            raise ValueError("Missing native method seed mapping")
        method[group][key] = seed[domain]
        destination = Path(output) / (domain + ".json")
        write(destination, method)
        config[domain]["config"] = str(destination)


def validate_frontend_handoff(plan_path, config_path, repo, root):
    from open3dbench.config import load_config
    from open3dbench.runner import FlowRunner
    from open3dbench.backend.contracts import validate_checkpoint
    proof = read(root.parent / "gpu_frontend.json")
    if proof["plan_sha256"] != sha(plan_path) or proof["config_sha256"] != sha(config_path):
        raise ValueError("GPU frontend plan/config changed")
    config = load_config(config_path)
    config["run_dir"] = str(root)
    runner = FlowRunner(repo, config, resume=True)
    task = read(read(plan_path)['task'])
    fixed = 'frontend_inputs' in task
    imported = set()
    if read(plan_path).get('incremental'):
        from .incremental import verify_prefix
        case = next(c for c in task['cases'] if c['id'] == read(plan_path)['case'])
        imported = set(verify_prefix(read(plan_path), task, case, root, config)['stages'])
    if fixed:
        from .frozen_frontend import verify
        case = next(c for c in task['cases'] if c['id'] == read(plan_path)['case'])
        verify(task, case, root, config)
    for stage, folder in (("input", "00_input"), ("floorplan", "10_floorplan"),
                          ("partition", "20_partition"), ("placement", "30_placement")):
        manifest = root / folder / "stage_manifest.json"
        if (sha(manifest) != proof["stage_manifests"][folder]
                or (stage not in imported and not (fixed and stage in {'input', 'floorplan'}) and not runner._resume_is_valid(stage))):
            raise ValueError("GPU frontend checkpoint is not signature/artifact identical: " + stage)
    validate_checkpoint(root / "30_placement/checkpoint.json", require_files=True)
    return proof


def execute(plan_path, *, prefix_only=False, phase=None):
    import yaml
    from open3dbench.pdks import get_pdk
    plan = read(plan_path)
    from .algorithm_scope import verify_runtime
    verify_runtime(plan)
    task = read(plan["task"])
    if validate(task) != plan["task_digest"]:
        raise ValueError("Task digest differs")
    if task["evaluator_sha256"] != evaluator_identity():
        raise ValueError("Evaluator differs from frozen task")
    case = next(c for c in task["cases"] if c["id"] == plan["case"])
    seed = next(s for s in task["seeds"] if s["id"] == plan["seed"]["id"])
    if seed != plan["seed"] or sha(plan["config"]) != case["config_sha256"]:
        raise ValueError("Case config/seed changed")
    output = Path(plan["output"])
    output.mkdir(exist_ok=True, parents=True)
    config_path = output / "resolved.yaml"
    split = plan.get("execution_mode") == "gpu-frontend-cpu-backend"
    if not prefix_only and split != (phase is not None):
        raise ValueError("Physical execution phase differs from the host plan")
    if phase != "backend":
        config = resolved_config(yaml.safe_load(Path(plan["config"]).read_text()), case, seed)
        bind_method_seeds(config, seed, output)
        if config_path.exists():
            raise ValueError("Evaluation attempt already exists")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    repo = Path(plan["source"])
    root = output / "run"
    prior = validate_frontend_handoff(plan_path, config_path, repo, root) if phase == "backend" else None
    commands = [[sys.executable, "-m", "open3dbench", "--repo", str(repo), "run", str(config_path),
                 "--run-dir", str(root), "--to", "floorplan" if prefix_only else ("placement" if phase == "frontend" else "cts")]]
    fixed = 'frontend_inputs' in task
    if plan.get('incremental'):
        commands = [[sys.executable, '-m', 'grt_bench.incremental', str(plan_path)]]
        if phase:
            commands[0] += ['--phase', phase]
    elif fixed:
        if prefix_only or not split or phase is None:
            raise ValueError('Fixed frontend requires phased execution; regeneration is forbidden')
        commands = [[sys.executable, '-m', 'grt_bench.frozen_frontend', str(plan_path), '--phase', phase]]
    if prefix_only:
        with (output / "prefix.log").open("x") as log:
            subprocess.run(commands[0], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=plan.get("timeout_seconds", 21600))
        from open3dbench.config import load_config
        from open3dbench.frontend_prefix import validate_prefix
        validate_prefix(repo, load_config(config_path), root)
        return
    if not fixed and phase == "backend":
        commands[0] += ["--from", "pdn", "--resume"]
    elif not fixed and plan.get("frontend_prefix"):
        commands[0] += ["--frontend-prefix", plan["frontend_prefix"]]
    if phase != "frontend":
        commands.append([sys.executable, "-m", "open3dbench", "--repo", str(repo), "global-route", str(config_path),
                         "--run-dir", str(root), "--output-dir", str(output / "grt")])
    elapsed = list(prior["commands"]) if prior else []
    for index, command in enumerate(commands):
        started = time.monotonic()
        with (output / ("flow-{}.log".format(index + (1 if prior else 0)))).open("x") as log:
            subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=plan.get("timeout_seconds", 21600))
        elapsed.append({"command": command, "wall_seconds": time.monotonic() - started})
    if phase == "frontend":
        write(output / "gpu_frontend.json", {"plan_sha256": sha(plan_path), "config_sha256": sha(config_path),
            "commands": elapsed, "stage_manifests": {folder: sha(root / folder / "stage_manifest.json")
                for folder in ("00_input", "10_floorplan", "20_partition", "30_placement")}})
        validate_frontend_handoff(plan_path, config_path, repo, root)
        return
    platform = repo / "OpenROAD-3D/flow/platforms" / (get_pdk(case["pdk"]).openroad_platform + "_3D")
    run(output / "grt/global_route_checkpoint.json", platform, Path(plan["hotspot"]), output / "analysis", endpoint=task["endpoint"])
    write(output / "execution.json", {"task_digest": plan["task_digest"], "case": case["id"], "seed": seed,
        "candidate": plan["candidate"], "commands": elapsed, "endpoint": task["endpoint"],
        "execution_mode": plan.get("execution_mode", "full"),
        "frontend_handoff_sha256": sha(output / "gpu_frontend.json") if split else None})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan", type=Path)
    p.add_argument("--prefix-only", action="store_true")
    p.add_argument("--phase", choices=("frontend", "backend"))
    a = p.parse_args()
    if a.prefix_only and a.phase:
        p.error("Prefix preparation cannot also be a physical phase")
    execute(a.plan, prefix_only=a.prefix_only, phase=a.phase)


if __name__ == "__main__":
    main()
