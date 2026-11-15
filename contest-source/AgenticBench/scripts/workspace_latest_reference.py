"""Versioned, read-only qualification for timing and thermal searches.

No flow/model launch occurs here. Publication needs a complete successful native
GRT run. Timing scores pre-CTS one-OPT STA; thermal scores post-CTS temperature
with a post-CTS STA penalty. Neither endpoint scores cached or routed timing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from local_bench import reference_latest as latest
from local_bench.reference_store import (adopt_reference, file_sha, load_reference,
    publish_reference, digest)
from scripts.workspace_task_selection import LATEST_PROFILE, load_registered_task
from scripts.workspace_task_selection import LATEST_THERMAL_PROFILE, LATEST_PROFILES, LATEST_TASKS

KIND = "latest-grt-timing-authority-v1"
THERMAL_KIND = "latest-cts-thermal-authority-v1"
FRAMEWORK = Path(__file__).resolve().parents[1]


def validate_latest_chain(run, *, reference_suffix=False):
    run = Path(run).resolve(strict=True)
    data = latest.read_json(run / "run.json")
    expected = list(latest.STAGE_DIRS)
    if not reference_suffix:
        expected.remove("cts")
    if data.get("status") != "success" or set(data.get("stages", {})) != set(expected):
        raise ValueError("Latest timing run has an incomplete or incompatible stage set")
    previous = None
    for stage in expected:
        if data["stages"][stage].get("status") != "success":
            raise ValueError("Latest stage did not succeed: " + stage)
        checked = latest.validate_stage(run, stage)
        manifest = checked["manifest"]
        upstream = "none" if previous is None else file_sha(run / latest.STAGE_DIRS[previous] / "stage_manifest.json")
        if manifest.get("signature", {}).get("upstream") != upstream:
            raise ValueError("Latest native upstream signature differs: " + stage)
        if stage in {"placement", "pdn", "hbt", "opt", "cts"}:
            root = run / latest.STAGE_DIRS[stage]
            checkpoint = latest.read_json(root / "checkpoint.json")
            if (checkpoint.get("schema_version") != 1 or checkpoint.get("stage") != stage
                    or checkpoint.get("pdk") != "ng45" or set(checkpoint.get("dies", {})) != {"bottom", "top"}):
                raise ValueError("Latest physical checkpoint identity differs: " + stage)
            if stage != "placement":
                pointer = Path(checkpoint.get("source_checkpoint", ""))
                if pointer.is_absolute() or (root / pointer).resolve() != (run / latest.STAGE_DIRS[previous] / "checkpoint.json").resolve():
                    raise ValueError("Latest physical checkpoint predecessor differs: " + stage)
            for record in checkpoint["dies"].values():
                for key in ("odb", "def", "verilog", "sdc"):
                    latest._safe_file(root, record[key], containment=run)
            for key in ("verilog", "sdc"):
                latest._safe_file(root, checkpoint["assembly"][key], containment=run)
            ready = {"hbt": "ready_for_opt", "opt": "ready_for_cts", "cts": "ready_for_route"}.get(stage)
            if ready and checkpoint.get(ready) is not True:
                raise ValueError("Latest checkpoint is not ready: " + stage)
        previous = stage
    return data


def normalized_recipe(case):
    """Exact recorded defaults of the pinned latest profile; no flow imports."""
    config = yaml.safe_load((latest.profile_path(case) / "config.yaml").read_text())
    config["floorplan"] = {"algorithm": "fixed"}
    config["pdn"] = {"algorithm": "openroad", "source_model": "independent-per-die"}
    config["pins"].update(default_die="bottom", owner_policy="fixed-bottom")
    config["placement"].update(density_policy="explicit", execution="direct")
    return config


def evaluator_digest():
    names = ["scripts/workspace_latest_reference.py", "scripts/workspace_codesign_evaluator.py",
             "scripts/workspace_framework_identity.py", "local_bench/reference_latest.py",
             "global_bench/verifiers/open3dbench_codesign_verifier.py", "scripts/workspace_cts_thermal.py",
             "scripts/workspace_runtime_image.py"]
    return digest({name: file_sha(FRAMEWORK / name) for name in names})


def contract_for(case, task, bundle):
    if task.objective.evaluator_profile not in LATEST_PROFILES:
        raise ValueError("Latest reference requires its versioned task/profile")
    checked = latest.validate_reference_bundle(bundle, case)
    config = yaml.safe_load((latest.profile_path(case) / "config.yaml").read_text())
    thermal = task.objective.evaluator_profile == LATEST_THERMAL_PROFILE
    return {"case": case, "endpoint": "cts_thermal" if thermal else "timing", "task_digest": task.digest(),
            "evaluator_digest": evaluator_digest(),
            "frozen_inputs": {"benchmark_input_manifest_sha256": file_sha(Path(bundle) / "benchmark_input.json"),
                              "physical_input_sha256": checked["physical_input_sha256"]},
            "physical": {"baseline_id": latest.BASELINE_ID, "profile": task.objective.evaluator_profile,
                         **({"scoring": {"primary_metric": task.objective.primary_metric,
                                         "direction": task.objective.direction,
                                         "parameters": dict(task.objective.parameters)}} if thermal else {}),
                         "source_revision": latest.SOURCE_REVISION, "image": latest.IMAGE_ID,
                         "native_revision": latest.NATIVE_REVISION, "opt_passes": 1,
                         "configuration": config,
                         **({"thermal_snapshot_threads": 4, "grid_size": 10, "ambient_k": 318.15,
                             "power_model": "continuous-passive-hbt-placement-estimated"} if thermal else {})}}


def _evidence(path, role):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path), "role": role}


def qualify_producer(producer):
    if isinstance(producer, dict) and producer.get("kind") == THERMAL_KIND:
        return qualify_thermal_producer(producer)
    if not isinstance(producer, dict) or producer.get("kind") != KIND:
        raise ValueError("Not a latest GRT timing authority")
    case, bundle = producer["case"], Path(producer["benchmark_input"])
    task = load_registered_task(producer["task_id"], objective="timing", framework=FRAMEWORK)
    contract = contract_for(case, task, bundle)
    manifest = latest.read_json(bundle / "benchmark_input.json")
    receipt_binding = manifest["latest_reference_receipt"]
    receipt_path = Path(receipt_binding["path"])
    receipt = latest.read_json(receipt_path)
    run = Path(producer["run_root"]).resolve(strict=True)
    if run != Path(receipt["source_run"]).resolve(strict=True):
        raise ValueError("Latest authority run is not the bound native source")
    if file_sha(run / "config.yaml") != receipt.get("native_config_sha256"):
        raise ValueError("Native reference config differs from its receipt")
    status = validate_latest_chain(run, reference_suffix=True)
    if status.get("design") != case or status.get("type") != "lol" or status.get("pdk") != "ng45":
        raise ValueError("Native reference case/integration/PDK mismatch")
    completion = latest.read_json(run / "reference-result.json")
    if (completion.get("status") != "success" or completion.get("endpoint") != "global-route"
            or completion.get("fresh_input") is not True or completion.get("source_commit") != latest.SOURCE_REVISION
            or completion.get("image_id") != latest.IMAGE_ID or completion.get("drt_dispatches") != 0
            or completion.get("executed_stages") != [*latest.STAGE_DIRS, "global_route"]
            or any(completion.get(k) != 0 for k in ("model_calls", "candidate_finals", "registry_publications"))):
        raise ValueError("Native reference completion/producer scope differs")
    config = yaml.safe_load((run / "config.yaml").read_text())
    expected = contract["physical"]["configuration"]
    # Paths may be rebound to the frozen physical source; their actual bytes
    # must still equal the trusted algorithm settings.
    rebound = copy.deepcopy(config)
    for domain in ("partition", "placement"):
        path = Path(rebound[domain]["config"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Native reference algorithm config is not a frozen file")
        trusted = FRAMEWORK / "local_bench/reference_configs" / (domain + ".json")
        if file_sha(path) != file_sha(trusted):
            raise ValueError("Native reference algorithm settings changed")
        rebound[domain]["config"] = expected[domain]["config"]
    normalized = normalized_recipe(case)
    for key in ("run_dir", "run_id"):
        rebound.pop(key, None)
        normalized.pop(key, None)
    if rebound != normalized:
        raise ValueError("Native reference scientific configuration differs")
    grt = latest.validate_grt_reference(run, config)
    if completion.get("grt_metrics") != grt["raw_metrics"]:
        raise ValueError("Native completion metrics differ from independent GRT artifacts")
    evidence = [_evidence(bundle / "benchmark_input.json", "input-bundle"),
                _evidence(receipt_path, "reference-import-receipt"),
                _evidence(run / "config.yaml", "native-config"), _evidence(run / "run.json", "native-run"),
                _evidence(run / "reference-result.json", "native-completion")]
    evidence.extend(_evidence(config[domain]["config"], "algorithm-config:" + domain)
                    for domain in ("partition", "placement"))
    runtime = receipt["runtime_identity"]
    evidence.append(_evidence(runtime["execution_proof"], "independent-execution-proof"))
    execution = latest.read_json(runtime["execution_proof"])
    evidence.extend(_evidence(record["path"], "execution-binding:" + key)
                    for key, record in execution["bindings"].items())
    if "equivalence_proof" in runtime:
        evidence.append(_evidence(runtime["equivalence_proof"], "independent-runtime-equivalence"))
    for stage, directory in latest.STAGE_DIRS.items():
        path = run / directory / "stage_manifest.json"
        if file_sha(path) != receipt.get("stage_manifest_sha256", {}).get(stage):
            raise ValueError("Native stage differs from sealed reference receipt: " + stage)
        data = latest.read_json(path)
        if stage in latest.STAGE_DIRS:
            identity = data.get("provenance", {})
            if (identity.get("repo_commit") != latest.SOURCE_REVISION
                    or identity.get("repo_dirty") is not False
                    or identity.get("container_image_id") != latest.IMAGE_ID):
                raise ValueError("Latest reference native stage producer mismatch: " + stage)
        evidence.append(_evidence(path, "stage-manifest:" + stage))
        evidence.extend(_evidence(run / directory / row["path"], "native-artifact:" + stage)
                        for row in data["artifacts"].values())
    for name in ("global_route_checkpoint.json", "grt_metrics.json"):
        evidence.append(_evidence(run / "80_route" / name, "grt-publication"))
    checkpoint = latest.read_json(run / "80_route/global_route_checkpoint.json")
    evidence.extend(_evidence(run / "80_route" / row.get("path", key), "grt-artifact")
                    for key, row in checkpoint["artifacts"].items())
    # Store evidence paths are unique even when manifests refer to shared files.
    evidence = list({row["path"]: row for row in evidence}.values())
    from global_bench.verifiers.open3dbench_codesign_verifier import _timing_row
    raw = _timing_row(run, expected_passes=1, latest=True, reference_suffix=True)
    return {"qualified": True, "case": case, "endpoint": "timing",
            "source_contract": contract["physical"], "frozen_inputs": contract["frozen_inputs"],
            "source_identity": {"source_revision": latest.SOURCE_REVISION, "image": latest.IMAGE_ID,
                                "native_revision": latest.NATIVE_REVISION},
            "baseline": {"source_run": str(run), "raw_metrics": raw}, "evidence": evidence,
            "historical_native_runtime": {k: row.get("runtime_seconds") for k, row in status["stages"].items()},
            "grt_qualification": grt, "signoff": False}


class LatestWorkspaceReference:
    @property
    def policy(self):
        """Latest endpoints only adopt qualified evidence, never run a fallback."""
        return "ADOPT"

    def __init__(self, runtime, case, objective, bundle, task):
        from scripts.workspace_framework_identity import revalidate_runtime_framework
        revalidate_runtime_framework(runtime)
        if (objective not in {"timing", "thermal"} or getattr(runtime, "reference_policy", "ADOPT") != "ADOPT"
                or runtime.source_commit != latest.SOURCE_REVISION):
            raise ValueError("Latest workspace uses ADOPT-only qualified endpoints with its exact computation source")
        latest.validate_reference_image(runtime.image)
        self.runtime, self.task, self.objective = runtime, task, objective
        self.contract = contract_for(case, task, bundle)
        self.record_path = Path(getattr(runtime, "reference_record", "") or "")
        if not self.record_path.is_absolute():
            raise ValueError("Latest workspace needs an explicit new qualified reference record")
        self.record_sha256 = file_sha(self.record_path)
        self.record = load_reference(self.record_path, contract=self.contract, validate=qualify_producer)

    @property
    def evaluator_invocations(self):
        return 0

    def to_dict(self):
        return {"policy": self.policy, "record": str(self.record_path), "record_sha256": self.record_sha256,
                "contract": self.contract, "fallback": "none", "reference_archive_endpoint": "global-route",
                "scoring_endpoint": "post-CTS-continuous-passive-power-HotSpot" if self.objective == "thermal" else "pre-CTS-one-OPT"}

    def obtain(self, evaluator, output):
        output = Path(output)
        if file_sha(self.record_path) != self.record_sha256:
            raise ValueError("Latest reference record changed")
        receipt = adopt_reference(self.record_path, contract=self.contract, validate=qualify_producer, output=output)
        from global_bench.verifiers.open3dbench_codesign_verifier import _timing_score_rows
        raw = receipt["qualification"]["baseline"]["raw_metrics"]
        from global_bench.verifiers.open3dbench_codesign_verifier import _timing_constrained_cts_thermal_score_rows
        metrics = _timing_constrained_cts_thermal_score_rows(raw, raw) if self.objective == "thermal" else _timing_score_rows(raw, raw)
        result = {"success": True, "metrics": metrics,
                  "run": receipt["qualification"]["baseline"]["source_run"],
                  "reference_origin": "adopted", "reference_id": receipt["reference_id"],
                  "reference_record": str(self.record_path), "reference_record_sha256": self.record_sha256,
                  "reference_evaluator_invocations": 0, "new_reference_native_attempts": 0,
                  "verification": str(output / "reference_adoption.json"),
                  "runtime_seconds": receipt["verification_wall_seconds"], "error": None,
                  "runtime_seconds_scope": "read-only reference validation, not native execution"}
        self.runtime.reference_record = self.record_path
        self.runtime.reference_record_sha256 = self.record_sha256
        self.runtime.reference_contract = self.contract
        self.runtime.reference_evaluation_root = Path(result["run"]).parent
        (output / "feedback.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def score_with_latest_record(candidate, reference, record_path, evaluation_contract_path, *, mode, profile, submission):
    from global_bench.verifiers.open3dbench_codesign_verifier import _timing_row, _timing_score_rows
    data = latest.read_json(evaluation_contract_path)
    thermal = profile == LATEST_THERMAL_PROFILE
    task = load_registered_task(submission["task_id"], objective="thermal" if thermal else "timing", framework=FRAMEWORK)
    if (mode != ("cts-thermal" if thermal else "timing") or profile not in LATEST_PROFILES or submission.get("task_digest") != task.digest()
            or data.get("task_digest") != task.digest() or data.get("source_revision") != latest.SOURCE_REVISION
            or data.get("image") != latest.IMAGE_ID or submission.get("base_revision") != latest.SOURCE_REVISION
            or data.get("evaluation_endpoint", {}).get("profile") != profile):
        raise ValueError("Latest candidate consumer differs from its immutable contract")
    contract = contract_for(data["case"], task, Path(data["benchmark_input"]))
    record = load_reference(record_path, contract=contract, validate=qualify_producer)
    if Path(record["qualification"]["baseline"]["source_run"]).resolve() != Path(reference).resolve():
        raise ValueError("Latest reference run mismatch")
    row = thermal_row(candidate, Path(candidate) / "95_thermal_snapshot") if thermal else _timing_row(candidate, expected_passes=1, latest=True)
    status = latest.read_json(Path(candidate) / "run.json")
    if status.get("design") != data["case"] or status.get("type") != "lol" or status.get("pdk") != "ng45":
        raise ValueError("Latest candidate case/integration/PDK mismatch")
    candidate_config = yaml.safe_load((Path(candidate) / "config.yaml").read_text())
    expected_config = normalized_recipe(data["case"])
    for key in ("run_dir", "run_id"):
        candidate_config.pop(key, None)
    if set(candidate_config) != set(expected_config):
        raise ValueError("Latest candidate configuration domains differ")
    for key in expected_config:
        if key not in {"partition", "placement"} and candidate_config[key] != expected_config[key]:
            raise ValueError("Latest candidate changed frozen physical configuration: " + key)
    for key in ("partition", "placement"):
        if candidate_config[key].get("seed") != expected_config[key]["seed"]:
            raise ValueError("Latest candidate changed a seed")
    if candidate_config["placement"].get("macro_halo_um") != expected_config["placement"]["macro_halo_um"]:
        raise ValueError("Latest candidate changed physical macro spacing")
    if thermal:
        from global_bench.verifiers.open3dbench_codesign_verifier import _timing_constrained_cts_thermal_score_rows
        return _timing_constrained_cts_thermal_score_rows(row, record["qualification"]["baseline"]["raw_metrics"])
    return _timing_score_rows(row, record["qualification"]["baseline"]["raw_metrics"])


def thermal_row(run, thermal_root):
    from scripts.workspace_cts_thermal import validate_snapshot
    from global_bench.verifiers.open3dbench_codesign_verifier import _timing_row, CTS_ALGORITHMS
    run = Path(run).resolve(strict=True)
    timing = _timing_row(run, expected_passes=1, latest=True, reference_suffix=True)
    cts = latest.read_json(run / "70_cts/metrics.json")
    if cts.get("status") != "success" or cts.get("algorithm") not in CTS_ALGORITHMS:
        raise ValueError("Thermal endpoint requires native CTS success")
    row = validate_snapshot(thermal_root, run / "70_cts/checkpoint.json")
    # Snapshot validation proves that this replay matches the immutable native
    # CTS STA. OPT timing above validates the upstream pass, not the penalty.
    cts_timing = _post_cts_timing(cts, Path(thermal_root) / "sta/metrics.tsv")
    return {**row,
            **cts_timing,
            "timing_evaluation_boundary": "post_cts_one_opt",
            "cts_algorithm": cts["algorithm"], "cut_net_count": timing["cut_net_count"],
            "boundary_budget_model": timing["boundary_budget_model"],
            "power_model": "continuous-passive-hbt-placement-estimated",
            "placement_total_hpwl_um": timing["placement_total_hpwl_um"]}


def _post_cts_timing(cts, replay_metrics):
    """Read the hash-validated CTS replay, cross-checking native CTS JSON."""
    from global_bench.verifiers.open3dbench_codesign_verifier import _finite
    required = {"graph_completeness": "passed", "unclocked_or_unconstrained_endpoints": 0,
                "hbt_is_passive_rc_in_authoritative_global_sta": True,
                "authoritative_timing_view": "continuous-passive-hbt",
                "global_timing_role": "post-cts-placement-estimated"}
    validity = cts.get("timing_validity", {})
    if any(validity.get(key) != value for key, value in required.items()):
        raise ValueError("Post-CTS timing validity gate failed")
    values = {}
    for line in Path(replay_metrics).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2 or fields[0] in values:
            raise ValueError("Malformed/duplicate post-CTS STA metric")
        values[fields[0]] = _finite(fields[1], fields[0])
    if values.get("passive_hbt_rc") != 1 or values.get("clock_count", 0) <= 0:
        raise ValueError("Post-CTS STA requires passive HBT RC and real clocks")
    native = cts.get("assembly", {}).get("timing", {})
    result = {}
    for metric in ("wns", "tns"):
        key = metric + "_setup"
        value = _finite(values.get(key), "post-CTS " + key)
        published = _finite(native.get(key), "CTS JSON " + key)
        if not math.isclose(value, published, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError("Post-CTS native/replay timing disagrees")
        result[key + "_ns"] = value
        result[metric + "_deficit"] = max(0.0, -value)
    return result


def qualify_thermal_producer(producer):
    """GRT publication proves the native prefix; new power evidence proves thermal."""
    case, bundle = producer["case"], Path(producer["benchmark_input"])
    task = load_registered_task(producer["task_id"], objective="thermal", framework=FRAMEWORK)
    parent = qualify_producer({"kind": KIND, "case": case, "benchmark_input": str(bundle),
                              "run_root": producer["run_root"], "task_id": LATEST_TASKS[10]})
    contract = contract_for(case, task, bundle)
    root = Path(producer["thermal_root"]).resolve(strict=True)
    raw = thermal_row(producer["run_root"], root)
    evidence = list(parent["evidence"])
    evidence.extend(_evidence(path, "post-cts-thermal") for path in sorted(root.rglob("*")) if path.is_file())
    return {**parent, "endpoint": "cts_thermal", "source_contract": contract["physical"],
            "frozen_inputs": contract["frozen_inputs"], "evidence": evidence,
            "baseline": {"source_run": producer["run_root"], "thermal_root": str(root), "raw_metrics": raw},
            "signoff": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=latest.CASES, required=True)
    parser.add_argument("--benchmark-input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--thermal-root", type=Path)
    args = parser.parse_args()
    task = load_registered_task(args.task_id, objective="thermal" if args.thermal_root else "timing", framework=FRAMEWORK)
    contract = contract_for(args.case, task, args.benchmark_input)
    producer = {"kind": KIND, "case": args.case, "task_id": task.task_id,
                "benchmark_input": str(args.benchmark_input.resolve()), "run_root": str(args.run.resolve())}
    if args.thermal_root:
        producer.update(kind=THERMAL_KIND, thermal_root=str(args.thermal_root.resolve()))
    path = publish_reference(args.store, contract=contract, origin="FRESH", producer=producer, validate=qualify_producer)
    print(json.dumps({"record": str(path), "sha256": file_sha(path), "native_dispatches": 0, "model_calls": 0}))


if __name__ == "__main__":
    main()
