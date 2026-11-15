"""Host-owned paired seeds for current local tasks; legacy defaults stay separate."""
from pathlib import Path

import yaml

from grt_bench.common import digest, read, sha
from .reference_latest import require


def validate_schedule(value):
    if not isinstance(value, dict) or set(value) != {"io", "partition", "placement", "routing"}:
        raise ValueError("Seeds require io, partition, placement and routing")
    if (any(type(v) is not int for v in value.values()) or value["io"] != 1
            or value["routing"] != -1
            or any(not 0 <= value[k] < 2**31 for k in ("partition", "placement"))):
        raise ValueError("Use fixed io=1, routing=-1 and nonnegative 31-bit frontend seeds")
    return dict(value)


def _artifact(run, directory, name):
    """Read only a hash-bound native artifact, not an unbound copied config."""
    root = Path(run) / directory
    manifest = read(root / "stage_manifest.json")
    entry = manifest["artifacts"][name]
    path = (root / entry["path"]).resolve(strict=True)
    path.relative_to(root.resolve())
    require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
            "Seed evidence changed: " + str(path))
    return read(path)


def validate_prefix(spec, run, config, seeds):
    from .evaluation import UPSTREAM_STAGE_DIRS
    require(run is not None, "Matrix cells need an immutable upstream input")
    identities = {}
    for stage in spec.upstream_stages:
        path = Path(run) / UPSTREAM_STAGE_DIRS[stage] / "stage_manifest.json"
        row = read(path)
        require(row.get("status") == "success" and
                (row.get("design"), row.get("integration"), row.get("pdk")) ==
                (config.get("design"), config.get("type"), config.get("pdk")),
                "Matrix input design/platform differs: " + stage)
        identities[stage] = sha(path)
    if "floorplan" in spec.upstream_stages:
        require(_artifact(run, "10_floorplan", "io_metrics")["seed"] == seeds["io"],
                "Frozen I/O seed differs")
    if "partition" in spec.upstream_stages:
        require(_artifact(run, "20_partition", "method_manifest")["seed"] == seeds["partition"],
                "Frozen partition seed differs; prepare a matching upstream input")
    if "placement" in spec.upstream_stages:
        # The root metrics bind the native params by checksum on maintained runs.
        # Placement's config signature also binds its seed. Reconstruct that
        # signature from the native run config; never trust the config alone.
        from open3dbench.runner import FlowRunner
        from open3dbench.config import load_config
        native_config = load_config(Path(run) / "config.yaml")
        runner = FlowRunner.__new__(FlowRunner)
        runner.config = native_config
        manifest = read(Path(run) / "30_placement/stage_manifest.json")
        require(digest(runner._stage_config("placement")) == manifest["signature"]["config"],
                "Frozen placement configuration is not bound to its manifest")
        require(native_config["placement"]["seed"] == seeds["placement"],
                "Frozen placement seed differs; prepare a matching upstream input")
    return identities


def validate_executed_seeds(spec, run, seeds):
    evidence = {}
    if spec.start_stage == "partition":
        path = Path(run) / "20_partition/partition_manifest.json"
        require(read(path)["seed"] == seeds["partition"], "Executed partition seed differs")
        evidence["partition"] = {"seed": seeds["partition"], "sha256": sha(path)}
    if spec.start_stage in {"partition", "placement"}:
        path = Path(run) / "30_placement/stages/02_global_place/params.json"
        params = read(path)
        require(params.get("random_seed") == seeds["placement"]
                and params.get("concurrent_lol", {}).get("initialization_seed") == seeds["placement"],
                "Executed placement seed differs")
        evidence["placement"] = {"seed": seeds["placement"], "sha256": sha(path)}
    return evidence


def cell_contract(spec, candidate, source, baseline, reference, base_config, seeds):
    """Bind new references to the exact seed vector, recipe, input and initial code."""
    from .program_bridge import materialize_program
    seeds = validate_schedule(seeds)
    config = yaml.safe_load(Path(base_config or spec.base_config).read_text())
    for key, stage in (("io", "pins"), ("partition", "partition"),
                       ("placement", "placement"), ("routing", "routing")):
        require(config.get(stage, {}).get("seed") == seeds[key],
                "Matrix base config seed differs: " + stage)
    initial = materialize_program(spec, spec.program_template.read_text(), source)
    contract = {"schema_version": 1, "task": spec.task_id, "seeds": seeds,
                "base_config_sha256": sha(base_config or spec.base_config),
                "initial_candidate_sha256": digest(initial),
                "seed_contract_sha256": sha(Path(__file__)),
                "upstream": validate_prefix(spec, baseline, config, seeds)}
    if reference is None:
        require(candidate == initial, "Only the initial program may qualify a matrix reference")
    else:
        receipt = read(Path(reference).parent / "evaluation_result.json")
        require(receipt.get("metrics", {}).get("runs_successfully") == 1
                and receipt.get("artifacts", {}).get("matrix_reference") is True
                and receipt["artifacts"].get("matrix_cell_contract") == contract,
                "Matrix reference task, source, seed, recipe or input differs; qualify it first")
    return contract
