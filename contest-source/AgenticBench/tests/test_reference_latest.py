"""No EDA/models: artifact mutation tests and real CLI plan/dry-run boundaries."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from local_bench import reference_latest as latest
from local_bench import reference_baseline as old
from local_bench.evaluation import evaluate_task
from local_bench.metrics import _source_grt_metrics_path, _grt_wirelength
from local_bench.model import load_task, load_candidate, resolve_task_path
from local_bench.program_bridge import initial_program_text, materialize_program

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "AgenticBench/local_tasks"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def record(path):
    return {"size_bytes": path.stat().st_size, "sha256": latest.sha256(path)}


def fixture(root):
    write(root / "70_cts/checkpoint.json", {"fixture": True})
    config = yaml.safe_load((latest.profile_path("bp_fe") / "config.yaml").read_text())
    route = root / "80_route"
    rows = {}
    for die in ("bottom", "top"):
        directory = route / die
        directory.mkdir(parents=True)
        for name in ("grt.tcl", "grt_state.tcl", "grt.guide", "global_route.odb", "hbt_ports.tsv",
                     "design.sdc", "design.sdc.audit.json", "design.sdc.ports.tsv"):
            (directory / name).write_text("fixture " + name)
        (directory / "grt.log").write_text("Total 100 50 50.0% 0 / 0 / 0\n"
            "[INFO GRT-0018] Total wirelength: 100.5 um\nGlobal routing runtime = 0:0:1\n"
            "OPEN3D_GRT_DONE die=" + die + "\n")
        observed = latest._native_grt_metrics(directory)
        write(directory / "grt_manifest.json", {"stage": "global_route", "status": "success",
              "signature": latest.sha256(directory / "grt.tcl"), "metrics": observed,
              "artifacts": {p.name: record(p) for p in directory.iterdir()}})
        rows[die] = dict(observed, runtime_seconds=2.0)
    metrics = {"stage": "global_route", "status": "success", "ready_for_detailed_route": True,
               "ready_for_extract": False, "dies": rows, "configuration": {
                   "threads": 4, "congestion_iterations": 4, "resource_reduction": 0.0},
               "grt_total_overflow": 0, "grt_wire_length_um": 201.0, "runtime_seconds": 4.0}
    write(route / "grt_metrics.json", metrics)
    checkpoint = {"stage": "global_route", "status": "success", "image_id": latest.IMAGE_ID,
        "ready_for_detailed_route": True, "ready_for_extract": False, "metrics": "grt_metrics.json",
        "source_checkpoint": "../70_cts/checkpoint.json",
        "source_checkpoint_sha256": latest.sha256(root / "70_cts/checkpoint.json"),
        "dies": {die: {"odb": die + "/global_route.odb", "guide": die + "/grt.guide",
                        "manifest": die + "/grt_manifest.json"} for die in rows},
        "artifacts": {name: record(route / name) for name in
                      ("grt_metrics.json", "bottom/grt_manifest.json", "top/grt_manifest.json")}}
    write(route / "global_route_checkpoint.json", checkpoint)
    return config


def full_reference(root):
    from scripts.workspace_latest_reference import normalized_recipe
    fixture(root)
    config = latest.bind_recipe_paths(normalized_recipe("bp_fe"), ROOT)
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    previous = None
    for stage, directory in latest.STAGE_DIRS.items():
        target = root / directory
        target.mkdir(exist_ok=True)
        (target / "design.def").write_text("fixture physical DEF")
        artifacts = {"def": {"path": "design.def", "bytes": (target / "design.def").stat().st_size,
                             "sha256": latest.sha256(target / "design.def")}}
        if stage == "partition":
            report = {"method": "tp-gnn-flat", "config": {"decoder": "global-2way"},
                      "hierarchy": {"mode": "flat", "consumed_as_feature": False}}
            write(target / "tp_gnn_report.json", report)
            write(target / "partition_manifest.json", {"method": "tp-gnn-flat", "method_report": report})
            for name, filename in (("method_manifest", "partition_manifest.json"), ("report", "tp_gnn_report.json")):
                path = target / filename
                artifacts[name] = {"path": filename, "bytes": path.stat().st_size, "sha256": latest.sha256(path)}
        if stage in ("placement", "pdn", "hbt", "opt", "cts"):
            checkpoint = {"schema_version": 1, "stage": stage, "pdk": "ng45",
                "source_checkpoint": "../" + latest.STAGE_DIRS[previous] + "/checkpoint.json",
                "dies": {d: {k: "design.def" for k in ("odb", "def", "verilog", "sdc")} for d in ("bottom", "top")},
                "assembly": {"verilog": "design.def", "sdc": "design.def"},
                "ready_for_opt": True, "ready_for_cts": True, "ready_for_route": True}
            write(target / "checkpoint.json", checkpoint)
            artifacts["checkpoint"] = {"path": "checkpoint.json", "bytes": (target / "checkpoint.json").stat().st_size,
                                       "sha256": latest.sha256(target / "checkpoint.json")}
        write(target / "stage_manifest.json", {"schema_version": 1, "stage": stage, "status": "success",
            "method": "tp-gnn-flat" if stage == "partition" else "fixture", "artifacts": artifacts,
            "signature": {"upstream": "none" if previous is None else latest.sha256(root / latest.STAGE_DIRS[previous] / "stage_manifest.json")},
            "provenance": {"repo_commit": latest.SOURCE_REVISION, "repo_dirty": False, "container_image_id": latest.IMAGE_ID}})
        previous = stage
    checkpoint = latest.read_json(root / "80_route/global_route_checkpoint.json")
    checkpoint["source_checkpoint_sha256"] = latest.sha256(root / "70_cts/checkpoint.json")
    write(root / "80_route/global_route_checkpoint.json", checkpoint)
    write(root / "run.json", {"design": "bp_fe", "type": "lol", "pdk": "ng45", "status": "success",
                              "stages": {s: {"status": "success"} for s in latest.STAGE_DIRS}})
    write(root / "reference-result.json", {"status": "success", "endpoint": "global-route", "fresh_input": True,
        "source_commit": latest.SOURCE_REVISION, "image_id": latest.IMAGE_ID, "actual_docker_image_id": latest.IMAGE_ID,
        "drt_dispatches": 0, "signoff": False, "executed_stages": list(latest.STAGE_DIRS) + ["global_route"]})


def execution_proof(root, run):
    bindings = {}
    for name in ("controller_plan", "dispatch_receipt", "container_result", "runtime_identity"):
        path = root / (name + ".json")
        write(path, {"synthetic_fixture": name})
        bindings[name] = {"path": str(path), "sha256": latest.sha256(path)}
    path = root / "execution-proof.json"
    write(path, {"schema_version": 1, "kind": "latest-reference-execution-proof-v1", "status": "verified",
        "case": "bp_fe", "source_run": str(run.resolve()), "source_revision": latest.SOURCE_REVISION,
        "native_revision": latest.NATIVE_REVISION, "sta_revision": latest.STA_REVISION,
        "canonical_image": latest.IMAGE_ID, "actual_docker_image_id": latest.IMAGE_ID,
        "binaries_sha256": latest.BINARY_SHA256, "container": {"id": "a" * 64,
            "state": "exited", "exit_code": 0, "oom_killed": False}, "bindings": bindings})
    return {"execution_proof": path, "execution_proof_sha256": latest.sha256(path)}


class LatestReferenceTests(unittest.TestCase):
    def test_full_import_is_new_and_preserves_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full_reference(root / "native")
            proof = execution_proof(root, root / "native")
            with patch("local_bench.reference_local_placement._source_revision", return_value=latest.SOURCE_REVISION):
                result = latest.adopt_latest_input(root / "native", root / "bundle", "bp_fe", ROOT, **proof)
                self.assertFalse(result["scoring_reference_qualified"])
                imported = latest.read_json(root / "bundle/run/run.json")
                self.assertEqual(imported["status"], "imported-prefix")
                self.assertEqual(set(imported["stages"]), {"input", "floorplan", "partition"})
                self.assertEqual(imported["reference_import"]["executed_stages"], [])
                self.assertEqual(latest.sha256(root / "bundle/run/config.yaml"),
                                 latest.sha256(root / "native/config.yaml"))
                for stage in ("input", "floorplan", "partition"):
                    name = latest.STAGE_DIRS[stage] + "/stage_manifest.json"
                    self.assertEqual(latest.sha256(root / "native" / name), latest.sha256(root / "bundle/run" / name))
                with self.assertRaises(ValueError):
                    latest.adopt_latest_input(root / "native", root / "bundle", "bp_fe", ROOT, **proof)
            ledger = root / "bundle/run/run.json"
            original = ledger.read_bytes()
            ledger.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Imported prefix ledger"):
                latest.validate_reference_bundle(root / "bundle", "bp_fe")
            ledger.write_bytes(original)
            (root / "native/run.json").write_text("changed")
            with self.assertRaises(ValueError):
                latest.validate_reference_bundle(root / "bundle", "bp_fe")

    def test_independent_execution_proof_is_required_and_revalidated(self):
        for mutation in ("missing", "native-failed", "oom", "binary", "binding"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                full_reference(root / "native")
                args = execution_proof(root, root / "native")
                path = args["execution_proof"]
                data = latest.read_json(path)
                if mutation == "missing":
                    args = {}
                elif mutation == "native-failed":
                    data["container"]["exit_code"] = 1
                elif mutation == "oom":
                    data["container"]["oom_killed"] = True
                elif mutation == "binary":
                    data["binaries_sha256"]["/usr/local/bin/sta"] = "0" * 64
                else:
                    (root / "controller_plan.json").write_text("changed")
                if args:
                    write(path, data)
                    args["execution_proof_sha256"] = latest.sha256(path)
                with patch("local_bench.reference_local_placement._source_revision", return_value=latest.SOURCE_REVISION):
                    with self.assertRaises(ValueError):
                        latest.adopt_latest_input(root / "native", root / "bundle", "bp_fe", ROOT, **args)
                self.assertFalse((root / "bundle").exists())

    def test_unknown_family_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root / "benchmark_input.json", {"baseline_id": "future-maybe"})
            with self.assertRaises(ValueError):
                latest.baseline_for_bundle(root)

    def test_explicit_image_families(self):
        latest.validate_reference_image(latest.IMAGE_ID)
        with self.assertRaises(ValueError):
            old.validate_reference_image(latest.IMAGE_ID)
        for image in old.IMAGE_IDS:
            with self.assertRaises(ValueError):
                latest.validate_reference_image(image)

    def test_all_profiles_exact_and_config_rebinding(self):
        for case in latest.CASES:
            with self.subTest(case=case):
                config = yaml.safe_load((latest.profile_path(case) / "config.yaml").read_text())
                latest.validate_reference_config(config, case)
                bound = latest.bind_recipe_paths(config, ROOT)
                self.assertEqual(bound["partition"]["config"], str(ROOT / latest.PARTITION_CONFIG))
                self.assertEqual(config["partition"]["config"], latest.PARTITION_CONFIG)
                changed = copy.deepcopy(config)
                changed["routing"]["congestion_iterations"] += 1
                with self.assertRaises(ValueError):
                    latest.validate_reference_config(changed, case)

    def test_native_grt_publication_and_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root)
            result = latest.validate_grt_reference(root, config)
            self.assertFalse(result["signoff"])
            self.assertEqual(_source_grt_metrics_path(root).name, "grt_metrics.json")
            self.assertEqual(_grt_wirelength(root, "bottom", result["raw_metrics"]["dies"]["bottom"]), 100.5)

    def test_grt_mutations_fail_closed(self):
        for change in ("artifact", "image", "budget", "handoff", "aggregate", "one-die", "drt"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = fixture(root)
                checkpoint_path = root / "80_route/global_route_checkpoint.json"
                checkpoint = latest.read_json(checkpoint_path)
                if change == "artifact":
                    (root / "80_route/top/grt.guide").write_text("tampered")
                elif change == "image":
                    checkpoint["image_id"] = next(iter(old.IMAGE_IDS))
                elif change == "budget":
                    config["routing"]["congestion_iterations"] += 1
                elif change == "handoff":
                    (root / "70_cts/checkpoint.json").write_text("tampered")
                elif change == "one-die":
                    del checkpoint["dies"]["top"]
                elif change == "drt":
                    checkpoint["ready_for_extract"] = True
                else:
                    path = root / "80_route/grt_metrics.json"
                    data = latest.read_json(path)
                    data["grt_wire_length_um"] = 0
                    write(path, data)
                    checkpoint["artifacts"]["grt_metrics.json"] = record(path)
                write(checkpoint_path, checkpoint)
                with self.assertRaises(ValueError):
                    latest.validate_grt_reference(root, config)

    def test_missing_native_metric_never_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "top"
            root.mkdir()
            (root / "grt.log").write_text("OPEN3D_GRT_DONE die=top\n")
            with self.assertRaises(ValueError):
                latest._native_grt_metrics(root)

    def test_safe_artifact_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            (Path(temp) / "outside").write_text("secret")
            (root / "alias").symlink_to("../outside")
            with self.assertRaises(ValueError):
                latest.validate_artifacts(root, {"alias": record(root / "alias")})

    def test_real_current_adapters_and_cli_dry_run(self):
        ids = (
               "placement_dynamic_parameters_reference_20260923",
               "global_routing_iteration_controller_source", "global_routing_maze_cost_source",
               "global_routing_ripup_policy_source")
        for task_id in ids:
            with self.subTest(task=task_id), tempfile.TemporaryDirectory() as temp:
                spec = load_task(resolve_task_path(TASKS, task_id))
                candidate = materialize_program(spec, initial_program_text(spec, load_candidate(spec.initial_candidate)), ROOT)
                folder = Path(temp)
                write(folder / "candidate/candidate.json", candidate)
                with patch.dict(os.environ, {"OPEN3DBENCH_PYTHON": sys.executable}):
                    result = evaluate_task(spec.root, folder / "candidate", ROOT, folder / "output",
                        baseline_run=folder / "unused-baseline", dry_run=True,
                        base_config=latest.profile_path("bp_fe") / "config.yaml")
                self.assertEqual(result.metrics.get("dry_run"), 1.0, result.to_dict())
                command = result.artifacts["command"]
                self.assertNotIn("timing_extract", command)
                if task_id.startswith("global_routing"):
                    self.assertIn("global-route", command)
                    self.assertNotIn("--to", command)
                    self.assertNotIn("open3dbench/runner.py", candidate["source_patch"])
                self.assertEqual(result.artifacts["native_flow_execution"], [])


if __name__ == "__main__":
    unittest.main()
