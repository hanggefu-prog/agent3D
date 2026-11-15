from __future__ import annotations

import json
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import yaml

from local_bench.metrics import extract_metrics
from local_bench.convergence import build_artifacts
from local_bench.model import load_candidate, load_task, resolve_task_path
from local_bench.program_bridge import initial_program_text
from local_bench.policy import validate_config_scope, validate_patch_text
from local_bench.verifier import verify_candidate
from local_bench.workspace import (
    WorkspaceError,
    _git_safe_directories,
    apply_candidate_patch,
    copy_upstream_stages,
    isolated_clone,
    materialize_dreamplace_source_tree,
    prepare_dreamplace_overlay,
    validate_candidate_workspace,
)
from local_bench.evaluation import (
    _discard_heavy_evaluation_artifacts,
    _deterministic_dreamplace_params,
    _dreamplace_global_placement_command,
    _materialize_frozen_inputs,
    _openroad_runtime_environment,
    _validate_routing_net_order_reference,
    _validate_dreamplace_global_output,
    _write_resolved_config,
    evaluate_task,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "local_tasks"


def _tree_identity(root):
    identity = {}
    for path in [root, *root.rglob("*")]:
        info = path.lstat()
        content = (os.readlink(path) if path.is_symlink() else
                   hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
        identity[str(path.relative_to(root))] = (info.st_ino, stat.S_IMODE(info.st_mode), content)
    return identity


@contextmanager
def _sealed_fixture(root):
    """Use real read-only fixtures, restoring only fixture modes for cleanup."""
    paths = [root, *root.rglob("*")]
    modes = {path: stat.S_IMODE(path.stat().st_mode) for path in paths if not path.is_symlink()}
    try:
        for path, mode in modes.items():
            path.chmod(mode & ~0o222)
        yield _tree_identity(root)
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


class TaskContractTest(unittest.TestCase):
    def test_retired_placement_launchers_are_absent(self) -> None:
        for name in ("run_placement_openevolve_case_task.sh", "launch_placement_openevolve_matrix.sh"):
            self.assertFalse((ROOT / "scripts" / name).exists())

    def test_deterministic_placement_params_do_not_mutate_frozen_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "run/10_floorplan/design_params.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps({"random_seed": 1000, "deterministic_flag": 0}) + "\n",
                encoding="utf-8",
            )
            derived = _deterministic_dreamplace_params(
                root / "run", root / "evaluation"
            )
            self.assertEqual(json.loads(source.read_text())["deterministic_flag"], 0)
            self.assertEqual(json.loads(derived.read_text())["deterministic_flag"], 1)

    def test_declared_frozen_input_is_copied_hashed_and_injected(self) -> None:
        spec = load_task(TASKS / "tp_gnn_post_placement_timing")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "baseline/20_partition/timing_input"
            source.mkdir(parents=True)
            (source / "timing_features.csv").write_text(
                "inst_name,worst_slack_ns\nu0,-0.1\n", encoding="utf-8"
            )
            (source / "timing_net_weights.csv").write_text(
                "net_index,weight\n0,5.0\n", encoding="utf-8"
            )
            (source / "timing_manifest.json").write_text(
                json.dumps({
                    "timing_status": "full_coverage_partition_last",
                    "timing_weight_mode": "paper_delay_slack_hyperedge_v1",
                }),
                encoding="utf-8",
            )
            (source / "provider.log").write_text("frozen provider\n", encoding="utf-8")
            (source / "openroad_full_partition_last.connections.tsv").write_text(
                "net_name\tdriver_inst\tsink_inst\tdelay_ns\tworst_slack_ns\n",
                encoding="utf-8",
            )
            output = root / "evaluation"
            config_path = output / "resolved_config.yaml"
            config_path.parent.mkdir(parents=True)
            resolved = {"partition": {"algorithm": "tp-gnn"}}
            config_path.write_text("partition:\n  algorithm: tp-gnn\n", encoding="utf-8")

            adopted = _materialize_frozen_inputs(
                spec,
                resolved,
                config_path,
                output,
                baseline_run=root / "baseline",
                reference_run=None,
            )

            destination = output / "harness_inputs/frozen/tp_gnn_timing"
            self.assertEqual(resolved["partition"]["timing_input"], str(destination))
            self.assertEqual(
                set(adopted["tp_gnn_timing"]["files"]),
                {
                    "provider.log",
                    "openroad_full_partition_last.connections.tsv",
                    "timing_features.csv",
                    "timing_manifest.json",
                    "timing_net_weights.csv",
                },
            )
            (source / "timing_net_weights.csv").write_text(
                "net_index,weight\n0,1.0\n", encoding="utf-8"
            )
            self.assertIn("0,5.0", (destination / "timing_net_weights.csv").read_text())

    def test_declared_frozen_input_rejects_wrong_timing_regime(self) -> None:
        spec = load_task(TASKS / "tp_gnn_post_placement_timing")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "baseline/20_partition/timing_input"
            source.mkdir(parents=True)
            for name in ("timing_features.csv", "timing_net_weights.csv"):
                (source / name).write_text("fixture\n", encoding="utf-8")
            (source / "openroad_full_partition_last.connections.tsv").write_text(
                "fixture\n", encoding="utf-8"
            )
            (source / "timing_manifest.json").write_text(
                json.dumps({"timing_status": "partial"}), encoding="utf-8"
            )
            output = root / "evaluation"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "assertion failed"):
                _materialize_frozen_inputs(
                    spec,
                    {"partition": {}},
                    output / "resolved_config.yaml",
                    output,
                    baseline_run=root / "baseline",
                    reference_run=None,
                )

    def test_openroad_runtime_ignores_an_incompatible_inherited_cpp_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / "libstdc++.so.6"
            library.write_text("fixture\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                ["c++"], 0, stdout=str(library) + "\n", stderr=""
            )
            with mock.patch("local_bench.evaluation.subprocess.run", return_value=completed):
                environment = _openroad_runtime_environment(
                    {"LD_LIBRARY_PATH": "/opt/conda/lib"}
                )
            self.assertEqual(
                environment["LD_LIBRARY_PATH"].split(":")[0],
                str(library.resolve().parent),
            )
            self.assertNotIn("/opt/conda/lib", environment["LD_LIBRARY_PATH"])

    def test_global_placement_boundary_calls_place3d_without_die_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "Open3DBench-V2.0"
            place = workspace / "Place-3D"
            flow_config = place / "candidate-placement.json"
            flow_config.parent.mkdir(parents=True)
            flow_config.write_text("{}\n", encoding="utf-8")
            default_config = (
                place
                / "config"
                / "placement"
                / "concurrent_dreamplace"
                / "default.json"
            )
            default_config.parent.mkdir(parents=True)
            default_config.write_text("{}\n", encoding="utf-8")
            run = root / "run"
            floorplan = run / "10_floorplan"
            partition = run / "20_partition"
            floorplan.mkdir(parents=True)
            partition.mkdir(parents=True)
            (floorplan / "design_params.json").write_text("{}\n", encoding="utf-8")
            (partition / "partition_assignment.csv").write_text(
                "inst_name,die\n", encoding="utf-8"
            )
            (partition / "die_regions.json").write_text("{}\n", encoding="utf-8")
            config = {
                "design": "bp_fe",
                "pdk": "ng45",
                "type": "lol",
                "input": {"variant": "default"},
                "placement": {
                    "algorithm": "concurrent-dreamplace",
                    "macro_halo_um": 10.0,
                    "config": "candidate-placement.json",
                    "gpu": True,
                    "target_density": 0.6,
                },
            }
            for directory in (
                "placement_parameters_reference_20260923",
                "placement_preconditioner",
            ):
                spec = load_task(TASKS / directory)
                task_config = dict(config)
                task_config["placement"] = dict(config["placement"])
                command, cwd = _dreamplace_global_placement_command(
                    spec, task_config, workspace, run, "python3"
                )
                self.assertEqual(cwd, place)
                self.assertIn("src.placement.concurrent_dreamplace", command)
                self.assertNotIn("open3dbench", command)
                self.assertNotIn("die_init", " ".join(command))
                self.assertIn(str(flow_config.resolve()), command)

    def test_global_placement_output_contract_rejects_die_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            stage = run / "30_placement" / "stages" / "02_global_place"
            stage.mkdir(parents=True)
            (stage / "stage_manifest.json").write_text(
                json.dumps({"returncode": 0}), encoding="utf-8"
            )
            for name in ("params.json", "run.log", "placement.gp.def"):
                (stage / name).write_text("complete\n", encoding="utf-8")
            outputs = _validate_dreamplace_global_output(run)
            self.assertIn("placement", outputs)
            (run / "30_placement" / "die_init").mkdir()
            with self.assertRaisesRegex(RuntimeError, "created die_init"):
                _validate_dreamplace_global_output(run)

    def test_copy_upstream_stages_rebases_run_owned_design_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            floorplan = root / "baseline" / "10_floorplan"
            floorplan.mkdir(parents=True)
            for name in ("design.def", "design.v", "design.sdc"):
                (floorplan / name).write_text(name + "\n", encoding="utf-8")
            (floorplan / "design_params.json").write_text(
                json.dumps(
                    {
                        "def_input": "../../old-run/10_floorplan/design.def",
                        "verilog_input": "../../old-run/10_floorplan/design.v",
                        "sdc_input": "../../old-run/10_floorplan/design.sdc",
                        "lef_input": ["/collateral/tech.lef"],
                    }
                ),
                encoding="utf-8",
            )

            run_dir = root / "candidate-run"
            copy_upstream_stages(root / "baseline", run_dir, ["floorplan"])

            copied = json.loads(
                (run_dir / "10_floorplan" / "design_params.json").read_text(
                    encoding="utf-8"
                )
            )
            for key, name in (
                ("def_input", "design.def"),
                ("verilog_input", "design.v"),
                ("sdc_input", "design.sdc"),
            ):
                self.assertEqual(
                    copied[key], str((run_dir / "10_floorplan" / name).resolve())
                )
            self.assertEqual(copied["lef_input"], ["/collateral/tech.lef"])

    def test_copy_upstream_can_link_only_large_frozen_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            floorplan = root / "baseline" / "10_floorplan"
            floorplan.mkdir(parents=True)
            large = floorplan / "design.def"
            large.write_bytes(b"x" * (1024 * 1024))
            for name in ("design.v", "design.sdc"):
                (floorplan / name).write_text(name + "\n", encoding="utf-8")
            params = floorplan / "design_params.json"
            params.write_text(
                json.dumps({"def_input": "/old/design.def"}), encoding="utf-8"
            )

            run_dir = root / "candidate-run"
            with mock.patch.dict(
                os.environ, {"OPEN3D_LINK_FROZEN_UPSTREAM": "1"}
            ):
                copy_upstream_stages(root / "baseline", run_dir, ["floorplan"])

            copied_large = run_dir / "10_floorplan" / "design.def"
            copied_params = run_dir / "10_floorplan" / "design_params.json"
            self.assertEqual(large.stat().st_ino, copied_large.stat().st_ino)
            self.assertNotEqual(params.stat().st_ino, copied_params.stat().st_ino)
            self.assertEqual(
                json.loads(copied_params.read_text(encoding="utf-8"))["def_input"],
                str(copied_large.resolve()),
            )

    def test_copy_sealed_upstream_keeps_source_modes_and_refreshes_only_copied_metadata(self):
        from local_bench.benchmark_input import _refresh_adopted_stage_manifest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline"
            stage = baseline / "10_floorplan"
            stage.mkdir(parents=True)
            (stage / "design.def").write_bytes(b"d" * (1024 * 1024))
            # Both mutable files exceed the hard-link threshold deliberately.
            (stage / "design_params.json").write_text(json.dumps({
                "def_input": "/old/design.def", "padding": "p" * (1024 * 1024),
            }))
            (stage / "stage_manifest.json").write_text(json.dumps({
                "artifacts": {"params": {"path": "design_params.json"}},
                "padding": "m" * (1024 * 1024),
            }))
            external = root / "external"
            external.mkdir()
            (external / "untouched.json").write_text("sealed\n")
            (stage / "linked-directory").symlink_to(external, target_is_directory=True)
            for link_mode in ("0", "1"):
                with self.subTest(link_mode=link_mode), _sealed_fixture(baseline) as before, \
                        _sealed_fixture(external) as external_before, mock.patch.dict(
                            os.environ, {"OPEN3D_LINK_FROZEN_UPSTREAM": link_mode}):
                    run = root / ("copy-" + link_mode)
                    copy_upstream_stages(baseline, run, ["floorplan"])
                    copied = run / "10_floorplan"
                    _refresh_adopted_stage_manifest(copied / "stage_manifest.json", baseline)
                    self.assertTrue(copied.stat().st_mode & stat.S_IWUSR)
                    for name in ("design_params.json", "stage_manifest.json"):
                        self.assertNotEqual((stage / name).stat().st_ino, (copied / name).stat().st_ino)
                        self.assertEqual((copied / name).stat().st_nlink, 1)
                        self.assertTrue((copied / name).stat().st_mode & stat.S_IWUSR)
                    self.assertFalse((copied / "design.def").stat().st_mode & stat.S_IWUSR)
                    self.assertEqual((stage / "design.def").stat().st_ino ==
                                     (copied / "design.def").stat().st_ino, link_mode == "1")
                    self.assertEqual(_tree_identity(baseline), before)
                    self.assertEqual(_tree_identity(external), external_before)
                    refreshed = json.loads((copied / "stage_manifest.json").read_text())
                    self.assertEqual(refreshed["artifacts"]["params"]["sha256"],
                                     hashlib.sha256((copied / "design_params.json").read_bytes()).hexdigest())

    def test_copy_upstream_rejects_symlinked_mutable_metadata_without_touching_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "baseline/10_floorplan"
            stage.mkdir(parents=True)
            external = root / "external"
            external.mkdir()
            (external / "params.json").write_text('{"def_input": "/old/design.def"}')
            (stage / "design.def").write_text("native\n")
            (stage / "design_params.json").symlink_to(external / "params.json")
            with _sealed_fixture(external) as before:
                with self.assertRaisesRegex(WorkspaceError, "must not be a symlink"):
                    copy_upstream_stages(root / "baseline", root / "copy", ["floorplan"])
                self.assertEqual(_tree_identity(external), before)

    def test_all_tasks_load(self):
        from local_bench.task_inventory import CORE_TASK_DIRECTORIES
        found = {p.name for p in TASKS.iterdir() if (p / "task.yaml").is_file()}
        self.assertEqual(found, set(CORE_TASK_DIRECTORIES))
        self.assertEqual(len({load_task(TASKS / name).task_id for name in found}), 10)

    def test_public_task_id_resolves_to_implementation_directory(self) -> None:
        self.assertEqual(
            resolve_task_path(TASKS, "placement_dynamic_parameters_reference_20260923"),
            (TASKS / "placement_parameters_reference_20260923").resolve(),
        )
        self.assertEqual(
            resolve_task_path(TASKS, "placement_preconditioner"),
            (TASKS / "placement_preconditioner").resolve(),
        )

    def test_routing_rejects_placement_override(self) -> None:
        spec = load_task(TASKS / "global_routing_ripup_policy")
        with self.assertRaisesRegex(ValueError, "outside this task scope"):
            validate_config_scope(spec, {"placement": {"target_density": 0.7}})

    def test_seeds_are_harness_owned_not_candidate_parameters(self):
        for task, stage in (("global_routing_ripup_policy", "routing"), ("tp_gnn_post_placement_timing", "partition")):
            spec = load_task(TASKS / task)
            candidate = load_candidate(spec.initial_candidate)
            candidate["config_overrides"].setdefault(stage, {})["seed"] = 7
            with self.assertRaisesRegex(ValueError, "outside this task scope"):
                verify_candidate(spec, candidate)

    def test_resolved_config_injects_common_random_numbers(self) -> None:
        spec = load_task(TASKS / "tp_gnn_post_placement_timing")
        candidate = load_candidate(spec.initial_candidate)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "resolved.yaml"
            resolved = _write_resolved_config(
                spec, spec.initial_candidate, output, candidate
            )
            self.assertEqual(resolved["pins"]["seed"], 1)
            self.assertEqual(resolved["partition"]["seed"], 1001)
            self.assertEqual(resolved["placement"]["seed"], 1000)
            self.assertEqual(resolved["routing"]["seed"], -1)
            # Current loss candidates cannot replace the frozen decoder recipe.
            self.assertEqual(resolved["partition"]["config"], "config/partition/tp_gnn_fast.json")
            self.assertEqual(yaml.safe_load(output.read_text()), resolved)


    def test_net_ordering_reference_is_bound_to_frozen_input_and_budget(self) -> None:
        spec = load_task(TASKS / "global_routing_ripup_policy")
        resolved = {
            "threads": 32,
            "routing": {
                "threads": 32,
                "congestion_iterations": 4,
                "resource_reduction": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root / "frozen"
            reference = root / "reference"
            stage_dirs = {
                "input": "00_input",
                "floorplan": "10_floorplan",
                "partition": "20_partition",
                "placement": "30_placement",
                "pdn": "40_pdn",
                "hbt": "50_hbt",
                "opt": "60_opt",
                "cts": "70_cts",
            }
            for stage in spec.upstream_stages:
                content = json.dumps({"stage": stage, "frozen": True}) + "\n"
                for run in (frozen, reference):
                    manifest = run / stage_dirs[stage] / "stage_manifest.json"
                    manifest.parent.mkdir(parents=True, exist_ok=True)
                    manifest.write_text(content, encoding="utf-8")
            route = reference / "80_route"
            route.mkdir()
            metrics = {
                "schema_version": 1,
                "stage": "route-grt",
                "status": "success",
                "algorithm": "openroad-die-local-net-order",
                "configuration": {
                    "threads": 32,
                    "congestion_iterations": 4,
                    "resource_reduction": 0.0,
                    "grt_only": True,
                },
                "dies": {
                    "bottom": {"status": "success"},
                    "top": {"status": "success"},
                },
            }
            (route / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            report = _validate_routing_net_order_reference(
                spec, frozen, reference, resolved
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["routing_budget"]["congestion_iterations"], 4)

            # Standalone GRT references publish grt_metrics.json with a
            # global-route checkpoint rather than legacy metrics.json.
            metrics["stage"] = "global_route"
            metrics.pop("algorithm")
            metrics["configuration"].pop("grt_only")
            (route / "metrics.json").unlink()
            (route / "grt_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (route / "global_route_checkpoint.json").write_text(json.dumps({
                "stage": "global_route", "status": "success",
                "metrics": "grt_metrics.json",
                "ready_for_detailed_route": True,
                "ready_for_extract": False,
            }), encoding="utf-8")
            report = _validate_routing_net_order_reference(
                spec, frozen, reference, resolved
            )
            self.assertEqual(report["status"], "passed")

            metrics["configuration"]["congestion_iterations"] = 3
            (route / "grt_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "budget mismatch"):
                _validate_routing_net_order_reference(
                    spec, frozen, reference, resolved
                )

            metrics["configuration"]["congestion_iterations"] = 4
            (route / "grt_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (reference / "70_cts" / "stage_manifest.json").write_text(
                '{"stage": "cts", "frozen": false}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "frozen cts stage"):
                _validate_routing_net_order_reference(
                    spec, frozen, reference, resolved
                )

    def test_source_patch_whitelist(self) -> None:
        spec = load_task(TASKS / "placement_preconditioner")
        good = """--- a/Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py
+++ b/Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py
@@ -1 +1 @@
-old
+new
"""
        report = validate_patch_text(spec, good)
        self.assertEqual(report["paths"], ["Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"])
        bad = good.replace("PlaceObj.py", "ElectricPotential.py")
        with self.assertRaisesRegex(ValueError, "outside this task scope"):
            validate_patch_text(spec, bad)

    def test_source_patch_rejects_file_deletion(self) -> None:
        spec = load_task(TASKS / "placement_preconditioner")
        patch = """diff --git a/Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py b/Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py
deleted file mode 100644
--- a/Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py
+++ /dev/null
@@ -1 +0,0 @@
-old
"""
        with self.assertRaisesRegex(ValueError, "creation, deletion"):
            validate_patch_text(spec, patch)

    def test_routing_resource_settings_are_not_candidate_parameters(self):
        spec = load_task(TASKS / "global_routing_ripup_policy")
        candidate = load_candidate(spec.initial_candidate)
        candidate["config_overrides"] = {"routing": {"threads": 0}}
        with self.assertRaisesRegex(ValueError, "outside this task scope"):
            verify_candidate(spec, candidate)

    def test_dynamic_placement_uses_a_task_adapter(self) -> None:
        spec = load_task(TASKS / "placement_parameters_reference_20260923")
        self.assertEqual(spec.program_mode, "task_adapter")
        self.assertEqual(spec.program_entrypoint, "adjust_learning_rate")
        self.assertEqual(spec.kind, "placement_dynamic_parameters")
        self.assertEqual(len(spec.allowed_source_paths), 2)
        self.assertEqual(
            spec.evaluation_boundary, "dreamplace-global-placement"
        )

    def test_both_placement_tasks_stop_at_native_global_placement(self) -> None:
        for directory in ("placement_parameters_reference_20260923", "placement_preconditioner"):
            spec = load_task(TASKS / directory)
            self.assertEqual(spec.start_stage, "placement")
            self.assertEqual(spec.stop_stage, "placement")
            self.assertEqual(
                spec.evaluation_boundary, "dreamplace-global-placement"
            )

    def test_tp_gnn_objective_uses_a_task_adapter(self) -> None:
        spec = load_task(TASKS / "tp_gnn_post_placement_timing")
        self.assertEqual(spec.program_mode, "task_adapter")
        self.assertEqual(spec.program_entrypoint, "compute_tp_gnn_loss")
        self.assertEqual(spec.kind, "tp_gnn_post_placement_timing")
        self.assertEqual(
            spec.allowed_source_paths,
            ["Place-3D/src/partition/lol/methods/tp_gnn.py"],
        )
        program = initial_program_text(spec, load_candidate(spec.initial_candidate))
        self.assertIn(
            "return -(\n"
            "        F.logsigmoid(positive_scores).sum()\n"
            "        + F.logsigmoid(-negative_scores).sum()\n"
            "    ) / max(1, source_batch_size)",
            program,
        )


class ConvergenceArtifactTest(unittest.TestCase):
    def test_history_generates_csv_json_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "order": 1,
                    "status": "accepted",
                    "score": 0.8,
                    "metrics": {
                        "total_object_origin_hpwl_um": 100.0,
                        "placement_overflow_max": 0.05,
                        "placement_max_density": 1.2,
                    },
                },
                {
                    "order": 2,
                    "status": "accepted",
                    "score": 0.9,
                    "metrics": {
                        "total_object_origin_hpwl_um": 90.0,
                        "placement_overflow_max": 0.04,
                        "placement_max_density": 1.1,
                    },
                },
            ]
            (root / "history.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = build_artifacts(root)
            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["best"]["order"], 2)
            self.assertEqual(report["best"]["placement_native_hpwl"], 90.0)
            self.assertTrue((root / "convergence.csv").is_file())
            self.assertIn("<svg", (root / "convergence.svg").read_text(encoding="utf-8"))


class MetricContractTest(unittest.TestCase):
    def test_extract_partition_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "20_partition"
            stage.mkdir()
            (stage / "partition_metrics.json").write_text(
                json.dumps({
                    "instance_count": 1000,
                    "top_count": 500,
                    "bottom_count": 500,
                    "cut_net_count": 100,
                    "cut_net_ratio": 0.1,
                    "cut_pin_count": 250,
                    "area_balance_error": 0.01,
                }),
                encoding="utf-8",
            )
            result = extract_metrics("partition", run, None)
            self.assertAlmostEqual(result["combined_score"], 0.875)
            self.assertEqual(result["partition_complete"], 1.0)

    def test_partition_imbalance_fails_constraint_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "20_partition"
            stage.mkdir()
            (stage / "partition_metrics.json").write_text(
                json.dumps({
                    "instance_count": 1000,
                    "top_count": 500,
                    "bottom_count": 500,
                    "cut_net_count": 10,
                    "cut_net_ratio": 0.01,
                    "cut_pin_count": 20,
                    "area_balance_error": 0.2,
                }),
                encoding="utf-8",
            )
            result = extract_metrics("partition", run, None)
            self.assertEqual(result["partition_complete"], 1.0)
            self.assertEqual(result["partition_constraints_passed"], 0.0)
            self.assertEqual(result["combined_score"], 0.0)

    def test_tp_gnn_objective_scores_final_cuts_against_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            candidate = root / "candidate"
            for run, cut_nets, cut_pins, timing_cut in (
                (reference, 100, 300, 125.0),
                (candidate, 80, 240, 100.0),
            ):
                stage = run / "20_partition"
                stage.mkdir(parents=True)
                (stage / "partition_metrics.json").write_text(
                    json.dumps({
                        "instance_count": 1000,
                        "top_count": 500,
                        "bottom_count": 500,
                        "macro_count": 20,
                        "top_macro_count": 10,
                        "cut_net_count": cut_nets,
                        "cut_net_ratio": cut_nets / 1000,
                        "cut_pin_count": cut_pins,
                        "timing_total_net_weight": 1000.0,
                        "timing_weighted_cut": timing_cut,
                        "timing_weighted_cut_ratio": timing_cut / 1000.0,
                        "timing_weight_mode": "paper_delay_slack_hyperedge_v1",
                        "area_balance_error": 0.02,
                        "method_metrics": {
                            "top_macro_area": "5000",
                            "bottom_macro_area": "5000",
                        },
                    }),
                    encoding="utf-8",
                )
            (candidate / "20_partition" / "tp_gnn_report.json").write_text(
                json.dumps({
                    "method": "tp-gnn",
                    "training": {
                        "status": "trained",
                        "final_loss": 1.25,
                        "epochs_completed": 100,
                        "optimizer_steps_completed": 600,
                        "final_embedding_diagnostics": {
                            "total_dimension_variance": 0.5,
                            "sample_pair_cosine_median": 0.25,
                            "training_pair_cosines": {
                                "population": "last_completed_training_batch", "embedding_state": "final",
                                "positive_sample_count": 10, "negative_sample_count": 20,
                                "positive_cosine_median": .8, "negative_cosine_median": .2,
                            },
                        },
                    },
                    "assignment_stages": {
                        "raw_weighted_kmeans": {"cut_net_count": 75, "timing_weighted_cut": 95.0},
                        "area_balanced_weighted_kmeans": {"cut_net_count": 78, "timing_weighted_cut": 98.0},
                        "lol_feasible_supernodes": {"cut_net_count": 80, "timing_weighted_cut": 100.0},
                        "raw_to_area_balanced_cut_delta": 3,
                        "area_balanced_to_feasible_cut_delta": 2,
                    },
                    "stage_timing_seconds": {"train_or_aggregate": 12.5},
                    "lol_handoff_contract": {"supernode_integrity": {
                        "preserved": True, "moved_group_count": 3, "moved_area_fraction": .125}},
                }),
                encoding="utf-8",
            )

            (reference / "20_partition" / "tp_gnn_report.json").write_text(
                json.dumps({"method": "tp-gnn", "config": {"decoder": "global-2way"}}),
                encoding="utf-8",
            )
            result = extract_metrics(
                "tp_gnn_learning_objective", candidate, reference
            )
            self.assertAlmostEqual(result["combined_score"], 1.25)
            self.assertAlmostEqual(result["timing_weighted_cut_score"], 1.25)
            self.assertEqual(
                result["timing_weight_mode"],
                "paper_delay_slack_hyperedge_v1",
            )
            self.assertEqual(result["partition_constraints_passed"], 1.0)
            self.assertEqual(result["tp_gnn_feasibility_cut_delta"], 2)
            self.assertEqual(result["tp_gnn_feasible_timing_weighted_cut"], 100.0)
            self.assertEqual(result["tp_gnn_epochs_completed"], 100)
            self.assertEqual(result["tp_gnn_area_balance_weighted_cut_delta"], 3)
            self.assertAlmostEqual(result["tp_gnn_feasibility_weighted_cut_relative_change"], 2/98)
            self.assertEqual(result["tp_gnn_repair_moved_supernodes"], 3)
            self.assertEqual(result["tp_gnn_repair_moved_area_fraction"], .125)
            self.assertEqual(result["tp_gnn_positive_pair_cosine_median"], .8)
            self.assertEqual(result["tp_gnn_negative_pair_sample_count"], 20)

            # Optional report data never turns missing measurements into zeros.
            report_path = candidate / "20_partition" / "tp_gnn_report.json"
            for optional in ({}, {
                "training": {"status": "trained", "final_loss": None,
                             "epochs_completed": "100", "optimizer_steps_completed": 1.5,
                             "final_embedding_diagnostics": {
                                 "total_dimension_variance": float("nan"),
                                 "sample_pair_cosine_median": True}},
                "assignment_stages": {
                    "raw_weighted_kmeans": {"cut_net_count": None},
                    "area_balanced_weighted_kmeans": None,
                    "lol_feasible_supernodes": {"timing_weighted_cut": float("inf")}},
                "stage_timing_seconds": {"train_or_aggregate": "12.5"},
            }):
                report_path.write_text(json.dumps({
                    "method": "tp-gnn", "training": {"status": "trained"}, **optional,
                }), encoding="utf-8")
                measured = extract_metrics("tp_gnn_learning_objective", candidate, reference)
                self.assertFalse(any(key.startswith("tp_gnn_") for key in measured))
                self.assertEqual(measured["combined_score"], result["combined_score"])
                self.assertEqual(measured["partition_constraints_passed"], 1.0)

            # Real zeros and signed closure deltas are meaningful measurements.
            report_path.write_text(json.dumps({
                "method": "tp-gnn",
                "training": {"status": "trained", "final_loss": 0.0,
                             "epochs_completed": 100,
                             "final_embedding_diagnostics": {
                                 "total_dimension_variance": 0.0,
                                 "sample_pair_cosine_median": None}},
                "assignment_stages": {
                    "raw_weighted_kmeans": {"cut_net_count": 0, "timing_weighted_cut": 0.0},
                    "area_balanced_weighted_kmeans": {"timing_weighted_cut": 0.0},
                    "lol_feasible_supernodes": {"timing_weighted_cut": 5.0},
                    "raw_to_area_balanced_cut_delta": -2,
                    "area_balanced_to_feasible_cut_delta": 0},
                "stage_timing_seconds": {"train_or_aggregate": 0.0},
            }), encoding="utf-8")
            measured = extract_metrics("tp_gnn_learning_objective", candidate, reference)
            self.assertEqual(measured["tp_gnn_final_loss"], 0.0)
            self.assertEqual(measured["tp_gnn_raw_cut_net_count"], 0)
            self.assertEqual(measured["tp_gnn_area_balance_cut_delta"], -2)
            self.assertEqual(measured["tp_gnn_feasibility_cut_delta"], 0)
            self.assertEqual(measured["tp_gnn_embedding_total_variance"], 0.0)
            self.assertEqual(measured["tp_gnn_training_runtime_seconds"], 0.0)
            self.assertNotIn("tp_gnn_embedding_pair_cosine_median", measured)
            self.assertNotIn("tp_gnn_balanced_cut_net_count", measured)
            self.assertEqual(measured["tp_gnn_feasibility_weighted_cut_delta"], 5.0)
            self.assertNotIn("tp_gnn_feasibility_weighted_cut_relative_change", measured)
            self.assertNotIn("tp_gnn_area_balance_weighted_cut_relative_change", measured)
            self.assertEqual(measured["combined_score"], result["combined_score"])

    def test_tp_gnn_objective_rejects_different_frozen_weight_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference/20_partition"
            candidate = root / "candidate/20_partition"
            reference.mkdir(parents=True)
            candidate.mkdir(parents=True)
            common = {
                "instance_count": 10,
                "top_count": 5,
                "bottom_count": 5,
                "macro_count": 0,
                "top_macro_count": 0,
                "cut_net_count": 1,
                "cut_net_ratio": 0.1,
                "cut_pin_count": 2,
                "timing_weighted_cut": 5.0,
                "timing_weighted_cut_ratio": 0.05,
                "timing_weight_mode": "paper_delay_slack_hyperedge_v1",
                "area_balance_error": 0.0,
            }
            (reference / "partition_metrics.json").write_text(
                json.dumps({**common, "timing_total_net_weight": 100.0}),
                encoding="utf-8",
            )
            (candidate / "partition_metrics.json").write_text(
                json.dumps({**common, "timing_total_net_weight": 200.0, "timing_weighted_cut_ratio": 0.025}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "same frozen timing net weights"):
                extract_metrics("tp_gnn_learning_objective", root / "candidate", root / "reference")

    def test_extract_routing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "80_route"
            stage.mkdir()
            (stage / "metrics.json").write_text(
                json.dumps({
                    "ready_for_extract": True,
                    "drc_violations": 0,
                    "antenna_violations": 0,
                    "runtime_seconds": 12.5,
                    "dies": {
                        "bottom": {"grt_total_overflow": 10, "grt_congestion_bins": 3},
                        "top": {"grt_total_overflow": 20, "grt_congestion_bins": 4},
                    },
                }),
                encoding="utf-8",
            )
            result = extract_metrics("routing_parameters", run, None)
            self.assertEqual(result["grt_total_overflow"], 30)
            self.assertEqual(result["grt_congestion_bins"], 7)
            self.assertEqual(result["routing_complete"], 1.0)

    def test_routing_antenna_violation_fails_correctness_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "80_route"
            stage.mkdir()
            (stage / "metrics.json").write_text(
                json.dumps({
                    "ready_for_extract": True,
                    "drc_violations": 0,
                    "antenna_violations": 1,
                    "dies": {
                        "bottom": {"status": "success", "grt_total_overflow": 0, "grt_congestion_bins": 0},
                        "top": {"status": "success", "grt_total_overflow": 0, "grt_congestion_bins": 0},
                    },
                }),
                encoding="utf-8",
            )
            result = extract_metrics("routing_parameters", run, None)
            self.assertEqual(result["routing_complete"], 0.0)
            self.assertEqual(result["combined_score"], 0.0)

    def test_net_ordering_score_uses_total_two_die_grt_overflow_and_wirelength(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            candidate = root / "candidate"
            for run, rows in (
                (reference, {"bottom": (12, 3, 100.0), "top": (18, 5, 200.0)}),
                (candidate, {"bottom": (6, 2, 120.0), "top": (14, 4, 180.0)}),
            ):
                stage = run / "80_route"
                stage.mkdir(parents=True)
                dies = {}
                for die, (overflow, bins, wirelength) in rows.items():
                    dies[die] = {
                        "status": "success",
                        "grt_total_overflow": overflow,
                        "grt_congestion_bins": bins,
                    }
                    die_dir = stage / die
                    die_dir.mkdir()
                    (die_dir / "route.log").write_text(
                        "[INFO GRT-0018] Total wirelength: {} um\n"
                        "OPEN3D_NET_ORDER die={} pass=0 phase=0.0 nets=100 "
                        "overflow_nets=20 hbt_nets=4 changed=40 changed_ratio=0.4 "
                        "overflow_changed=10 overflow_changed_ratio=0.5 "
                        "top10_overlap=0.7 score_min=0.0 score_max=1.2 "
                        "score_mean=0.6 score_std=0.2 baseline_hash=11 learned_hash=22\n".format(
                            wirelength, die
                        ),
                        encoding="utf-8",
                    )
                # No extraction, DRC, or antenna fields are required by this task.
                (stage / "metrics.json").write_text(
                    json.dumps({"dies": dies, "runtime_seconds": 7.0}),
                    encoding="utf-8",
                )
            result = extract_metrics(
                "routing_net_ordering_source", candidate, reference
            )
            self.assertEqual(result["grt_total_overflow"], 20)
            self.assertEqual(result["grt_wirelength_um"], 300.0)
            self.assertEqual(result["bottom_grt_total_overflow"], 6)
            self.assertEqual(result["top_grt_total_overflow"], 14)
            self.assertAlmostEqual(result["overflow_score"], 1.5)
            self.assertAlmostEqual(result["wirelength_score"], 1.0)
            self.assertAlmostEqual(result["combined_score"], 1.05)
            self.assertEqual(result["grt_complete"], 1.0)
            self.assertEqual(result["net_order_diagnostics_present"], 1.0)
            self.assertEqual(result["net_order_calls"], 2.0)
            self.assertEqual(result["net_order_changed_calls"], 2.0)
            self.assertAlmostEqual(result["net_order_changed_ratio_max"], 0.4)
            self.assertAlmostEqual(
                result["net_order_overflow_changed_ratio_max"], 0.5
            )
            self.assertAlmostEqual(result["net_order_top10_overlap_min"], 0.7)

    def test_net_ordering_score_allows_cross_die_tradeoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for name, bottom, top in (("balanced", 10, 10), ("traded", 2, 18)):
                run = root / name
                stage = run / "80_route"
                stage.mkdir(parents=True)
                dies = {}
                for die, overflow, wirelength in (
                    ("bottom", bottom, 100.0),
                    ("top", top, 200.0),
                ):
                    dies[die] = {
                        "status": "success",
                        "grt_total_overflow": overflow,
                        "grt_congestion_bins": overflow,
                        "grt_wirelength_um": wirelength,
                    }
                (stage / "metrics.json").write_text(
                    json.dumps({"dies": dies}), encoding="utf-8"
                )
                runs.append(run)
            balanced = extract_metrics("routing_net_ordering_source", runs[0], None)
            traded = extract_metrics("routing_net_ordering_source", runs[1], None)
            self.assertEqual(balanced["combined_score"], traded["combined_score"])
            self.assertEqual(balanced["grt_complete"], 0.0)

    def test_net_ordering_missing_hook_diagnostics_is_not_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "80_route"
            stage.mkdir(parents=True)
            dies = {}
            for die in ("bottom", "top"):
                dies[die] = {
                    "status": "success",
                    "grt_total_overflow": 10,
                    "grt_congestion_bins": 2,
                    "grt_wirelength_um": 100.0,
                }
            (stage / "metrics.json").write_text(
                json.dumps({"dies": dies}), encoding="utf-8"
            )
            result = extract_metrics("routing_net_ordering_source", run, None)
            self.assertEqual(result["net_order_diagnostics_present"], 0.0)
            self.assertEqual(result["grt_complete"], 0.0)
            self.assertEqual(result["combined_score"], 0.0)

    def test_net_ordering_accepts_one_die_that_converges_before_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "80_route"
            stage.mkdir(parents=True)
            dies = {}
            for die in ("bottom", "top"):
                dies[die] = {
                    "status": "success",
                    "grt_total_overflow": 0,
                    "grt_congestion_bins": 0,
                    "grt_wirelength_um": 100.0,
                }
                die_dir = stage / die
                die_dir.mkdir()
                log = "[INFO GRT-0018] Total wirelength: 100 um\n"
                if die == "bottom":
                    log += (
                        "OPEN3D_NET_ORDER die=bottom pass=0 phase=0 nets=100 "
                        "overflow_nets=2 hbt_nets=4 changed=0 changed_ratio=0 "
                        "overflow_changed=0 overflow_changed_ratio=0 "
                        "top10_overlap=1 score_min=0 score_max=1 "
                        "score_mean=0.5 score_std=0.2 baseline_hash=11 "
                        "learned_hash=11\n"
                    )
                (die_dir / "route.log").write_text(log, encoding="utf-8")
            (stage / "metrics.json").write_text(
                json.dumps({"dies": dies}), encoding="utf-8"
            )

            result = extract_metrics("routing_net_ordering_source", run, None)

            self.assertEqual(result["net_order_diagnostics_present"], 1.0)
            self.assertEqual(result["bottom_net_order_calls"], 1.0)
            self.assertEqual(result["top_net_order_calls"], 0.0)
            self.assertEqual(result["grt_complete"], 1.0)
            self.assertGreater(result["combined_score"], 0.0)

    def test_active_routing_source_tasks_use_smooth_total_die_objective(self) -> None:
        prefixes = {
            "routing_ripup_policy_source": (
                "OPEN3D_RIPUP_POLICY die={die} calls=20 changed=5 "
                "changed_ratio=0.25 selected=9 selected_ratio=0.45 score=0.7"
            ),
            "routing_maze_cost_source": (
                "OPEN3D_MAZE_COST die={die} calls=200 changed=100 "
                "changed_ratio=0.5 multiplier_mean=1.1 multiplier_last=1.2"
            ),
            "routing_iteration_controller_source": (
                "OPEN3D_ITERATION_CONTROL die={die} calls=2 changed=1 "
                "cost_scale=1.2 search_scale=1.1 ripup_offset=1.0 "
                "cost_height=20 search_window=12 ripup_threshold=0"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            reference_stage = reference / "80_route"
            reference_stage.mkdir(parents=True)
            reference_dies = {
                "bottom": {
                    "status": "success",
                    "grt_total_overflow": 8,
                    "grt_congestion_bins": 3,
                    "grt_wirelength_um": 100.0,
                },
                "top": {
                    "status": "success",
                    "grt_total_overflow": 12,
                    "grt_congestion_bins": 4,
                    "grt_wirelength_um": 200.0,
                },
            }
            (reference_stage / "metrics.json").write_text(
                json.dumps({"dies": reference_dies}), encoding="utf-8"
            )

            expected_overflow_cost = 1010.0 / 1020.0
            expected_wirelength_cost = 285.0 / 300.0
            expected_score = 1.0 / (
                0.1 * expected_overflow_cost + 0.9 * expected_wirelength_cost
            )
            for kind, diagnostic in prefixes.items():
                candidate = root / kind
                stage = candidate / "80_route"
                stage.mkdir(parents=True)
                dies = {
                    "bottom": {
                        "status": "success",
                        "grt_total_overflow": 4,
                        "grt_congestion_bins": 2,
                        "grt_wirelength_um": 95.0,
                    },
                    "top": {
                        "status": "success",
                        "grt_total_overflow": 6,
                        "grt_congestion_bins": 3,
                        "grt_wirelength_um": 190.0,
                    },
                }
                for die in ("bottom", "top"):
                    die_dir = stage / die
                    die_dir.mkdir()
                    (die_dir / "route.log").write_text(
                        diagnostic.format(die=die) + "\n", encoding="utf-8"
                    )
                (stage / "metrics.json").write_text(
                    json.dumps({"dies": dies, "runtime_seconds": 3.0}),
                    encoding="utf-8",
                )
                result = extract_metrics(kind, candidate, reference)
                self.assertAlmostEqual(result["overflow_cost"], expected_overflow_cost)
                self.assertAlmostEqual(result["wirelength_cost"], expected_wirelength_cost)
                self.assertAlmostEqual(result["combined_score"], expected_score)
                self.assertEqual(result["policy_diagnostics_present"], 1.0)
                self.assertGreater(result["policy_calls"], 0.0)
                self.assertEqual(result["grt_total_overflow"], 10)
                self.assertEqual(result["grt_wirelength_um"], 285.0)

    def test_extract_placement_overflow_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "30_placement"
            native = stage / "stages" / "02_global_place"
            native.mkdir(parents=True)
            (native / "run.log").write_text(
                "iteration 3999, wHPWL 2.000000E+06, "
                "Overflow [2.000E-02, 3.000E-02], "
                "MaxDensity [1.02, 1.04]\n",
                encoding="utf-8",
            )
            result = extract_metrics("placement_parameters", run, None)
            self.assertEqual(result["placement_native_hpwl"], 2_000_000)
            self.assertAlmostEqual(result["placement_overflow_max"], 0.03)
            self.assertAlmostEqual(result["placement_overflow_mean"], 0.025)
            self.assertAlmostEqual(result["placement_max_density"], 1.04)

    def test_placement_score_uses_nine_to_one_hpwl_overflow_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            candidate = root / "candidate"
            for run, hpwl, overflow in (
                (reference, 100.0, 0.1),
                (candidate, 80.0, 0.2),
            ):
                stage = run / "30_placement"
                native = stage / "stages" / "02_global_place"
                native.mkdir(parents=True)
                (native / "run.log").write_text(
                    "iteration 1, wHPWL {:.6E}, Overflow {:.6f}, "
                    "MaxDensity 1.0\n".format(hpwl, overflow),
                    encoding="utf-8",
                )

            result = extract_metrics("placement_parameters", candidate, reference)
            self.assertAlmostEqual(result["hpwl_score"], 1.25)
            self.assertAlmostEqual(result["overflow_score"], 0.5)
            self.assertAlmostEqual(result["combined_score"], 1.175)

    def test_placement_score_does_not_clip_strong_improvements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            candidate = root / "candidate"
            for run, hpwl, overflow in (
                (reference, 100.0, 0.1),
                (candidate, 20.0, 0.01),
            ):
                native = run / "30_placement" / "stages" / "02_global_place"
                native.mkdir(parents=True)
                (native / "run.log").write_text(
                    "iteration 1, wHPWL {:.6E}, Overflow {:.6f}, MaxDensity 1.0\n".format(
                        hpwl, overflow
                    ),
                    encoding="utf-8",
                )
            result = extract_metrics("placement_parameters", candidate, reference)
            self.assertAlmostEqual(result["hpwl_score"], 5.0)
            self.assertAlmostEqual(result["overflow_score"], 10.0)
            self.assertAlmostEqual(result["combined_score"], 5.5)

    def test_missing_placement_density_log_is_not_scored_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "30_placement"
            stage.mkdir()
            with self.assertRaisesRegex(ValueError, "run.log is missing"):
                extract_metrics("placement_parameters", run, None)

    def test_dynamic_placement_requires_and_reports_controller_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stage = run / "30_placement"
            native = stage / "stages" / "02_global_place"
            native.mkdir(parents=True)
            (native / "run.log").write_text(
                "DynamicLR step=0 base_lr=0.01 bottom_lr=0.01 top_lr=0.01 "
                "bottom_multiplier=1 top_multiplier=1 hpwl=2e6 "
                "overflow_bottom=0.2 overflow_top=0.18\n"
                "DynamicLR step=1 base_lr=0.009 bottom_lr=0.0099 top_lr=0.0081818181818 "
                "bottom_multiplier=1.1 top_multiplier=0.90909090909 hpwl=1.9e6 "
                "overflow_bottom=0.1 overflow_top=0.08\n"
                "iteration 1, wHPWL 1.900000E+06, "
                "Overflow [1.000E-01, 8.000E-02], "
                "MaxDensity [1.1, 1.08]\n",
                encoding="utf-8",
            )
            result = extract_metrics("placement_dynamic_parameters", run, None)
            self.assertEqual(result["dynamic_lr_updates"], 2)
            self.assertEqual(result["dynamic_lr_final"], 0.009)
            self.assertAlmostEqual(result["dynamic_lr_bottom_final"], 0.0099)
            self.assertAlmostEqual(result["dynamic_lr_top_final"], 0.0081818181818)
            self.assertLessEqual(result["dynamic_lr_max_die_ratio"], 4.0)


class RuntimeOverlayTest(unittest.TestCase):
    def test_compaction_retains_placement_evidence_before_deleting_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stage = output / "run/30_placement/stages/02_global_place"
            stage.mkdir(parents=True)
            params = stage / "params.json"
            manifest = stage / "stage_manifest.json"
            run_log = stage / "run.log"
            params.write_text('{"optimizer": "nesterov"}\n', encoding="utf-8")
            manifest.write_text('{"status": "completed"}\n', encoding="utf-8")
            run_log.write_text("prefix\nfinal placement metrics\n", encoding="utf-8")
            (output / "workspace").mkdir()
            (output / "runtime").mkdir()
            artifacts = {
                "global_placement_outputs": {
                    "params": str(params),
                    "manifest": str(manifest),
                    "log": str(run_log),
                }
            }

            _discard_heavy_evaluation_artifacts(output, artifacts)

            self.assertFalse((output / "run").exists())
            self.assertFalse((output / "workspace").exists())
            self.assertFalse((output / "runtime").exists())
            retained = output / "retained_artifacts"
            self.assertEqual(
                (retained / "placement_params.json").read_text(encoding="utf-8"),
                '{"optimizer": "nesterov"}\n',
            )
            self.assertEqual(
                (retained / "placement_stage_manifest.json").read_text(
                    encoding="utf-8"
                ),
                '{"status": "completed"}\n',
            )
            self.assertIn(
                "final placement metrics",
                (retained / "placement_run_tail.log").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                artifacts["global_placement_outputs"]["log"],
                str(retained / "placement_run_tail.log"),
            )

    def test_vendored_openroad_patch_stays_in_the_outer_clone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            utility = (
                source
                / "OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp"
            )
            utility.parent.mkdir(parents=True)
            utility.write_text("int order = 1;\n", encoding="utf-8")
            route = source / "open3dbench/backend/three_d_route.py"
            route.parent.mkdir(parents=True)
            route.write_text("MODE = 'full'\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=source, check=True)
            subprocess.run(["git", "add", "."], cwd=source, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Test", "-c",
                    "user.email=test@example.com", "commit", "-qm", "baseline",
                ],
                cwd=source,
                check=True,
            )

            workspace = root / "workspace"
            isolated_clone(
                source,
                workspace,
                mutable_submodules=["OpenROAD-3D/openroad"],
            )
            self.assertFalse(
                (workspace / "OpenROAD-3D/openroad/.git").exists()
            )
            patch = root / "candidate.patch"
            patch.write_text(
                "--- a/OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp\n"
                "+++ b/OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp\n"
                "@@ -1 +1 @@\n"
                "-int order = 1;\n"
                "+int order = 2;\n"
                "--- a/open3dbench/backend/three_d_route.py\n"
                "+++ b/open3dbench/backend/three_d_route.py\n"
                "@@ -1 +1 @@\n"
                "-MODE = 'full'\n"
                "+MODE = 'grt'\n",
                encoding="utf-8",
            )

            result = apply_candidate_patch(workspace, patch)

            self.assertEqual(result["status"], "applied")
            self.assertEqual(
                (
                    workspace
                    / "OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp"
                ).read_text(encoding="utf-8"),
                "int order = 2;\n",
            )
            self.assertEqual(
                (workspace / "open3dbench/backend/three_d_route.py").read_text(
                    encoding="utf-8"
                ),
                "MODE = 'grt'\n",
            )
            self.assertEqual(utility.read_text(encoding="utf-8"), "int order = 1;\n")

    def test_nested_openroad_patch_uses_an_isolated_mutable_clone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            nested = source / "OpenROAD-3D" / "openroad"
            utility = nested / "src/grt/src/fastroute/src/utility.cpp"
            utility.parent.mkdir(parents=True)
            utility.write_text("int order = 1;\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=nested, check=True)
            subprocess.run(["git", "add", "."], cwd=nested, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Test", "-c",
                    "user.email=test@example.com", "commit", "-qm", "nested baseline",
                ],
                cwd=nested,
                check=True,
            )
            route = source / "open3dbench/backend/three_d_route.py"
            route.parent.mkdir(parents=True)
            route.write_text("MODE = 'full'\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=source, check=True)
            subprocess.run(["git", "add", "."], cwd=source, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Test", "-c",
                    "user.email=test@example.com", "commit", "-qm", "outer baseline",
                ],
                cwd=source,
                check=True,
            )

            workspace = root / "workspace"
            isolated_clone(
                source,
                workspace,
                mutable_submodules=["OpenROAD-3D/openroad"],
            )
            patch = root / "candidate.patch"
            patch.write_text(
                "--- a/OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp\n"
                "+++ b/OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp\n"
                "@@ -1 +1 @@\n"
                "-int order = 1;\n"
                "+int order = 2;\n"
                "--- a/open3dbench/backend/three_d_route.py\n"
                "+++ b/open3dbench/backend/three_d_route.py\n"
                "@@ -1 +1 @@\n"
                "-MODE = 'full'\n"
                "+MODE = 'grt'\n",
                encoding="utf-8",
            )
            result = apply_candidate_patch(workspace, patch)
            self.assertEqual(result["status"], "applied")
            self.assertEqual(
                (
                    workspace
                    / "OpenROAD-3D/openroad/src/grt/src/fastroute/src/utility.cpp"
                ).read_text(encoding="utf-8"),
                "int order = 2;\n",
            )
            self.assertEqual(
                (workspace / "open3dbench/backend/three_d_route.py").read_text(
                    encoding="utf-8"
                ),
                "MODE = 'grt'\n",
            )
            self.assertEqual(utility.read_text(encoding="utf-8"), "int order = 1;\n")
            self.assertEqual(route.read_text(encoding="utf-8"), "MODE = 'full'\n")

    def test_isolated_clone_trusts_only_the_mounted_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "tracked.txt").write_text("content\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=OpenAgentic3DBench",
                    "-c",
                    "user.email=benchmark@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=source,
                check=True,
            )

            clone = isolated_clone(source, root / "clone")

            self.assertEqual((clone / "tracked.txt").read_text(encoding="utf-8"), "content\n")
            self.assertEqual(list(root.glob(".openagentic3d-gitconfig-*")), [])

    def test_manual_isolated_clone_can_snapshot_current_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "modified.txt").write_text("committed\n", encoding="utf-8")
            (source / "deleted.txt").write_text("delete me\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=source, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=OpenAgentic3DBench", "-c",
                    "user.email=benchmark@example.invalid", "commit", "-qm", "fixture",
                ],
                cwd=source,
                check=True,
            )
            (source / "modified.txt").write_text("working tree\n", encoding="utf-8")
            (source / "untracked.txt").write_text("new file\n", encoding="utf-8")
            (source / "deleted.txt").unlink()

            clone = isolated_clone(
                source, root / "clone", include_worktree=True
            )

            self.assertEqual(
                (clone / "modified.txt").read_text(encoding="utf-8"),
                "working tree\n",
            )
            self.assertEqual(
                (clone / "untracked.txt").read_text(encoding="utf-8"),
                "new file\n",
            )
            self.assertFalse((clone / "deleted.txt").exists())

    def test_isolated_clone_resolves_a_worktree_gitdir_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            worktree = root / "worktree"
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "tracked.txt").write_text("content\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=OpenAgentic3DBench", "-c",
                    "user.email=benchmark@example.invalid", "commit", "-qm", "fixture",
                ],
                cwd=source,
                check=True,
            )
            subprocess.run(
                ["git", "worktree", "add", "-q", "--detach", str(worktree)],
                cwd=source,
                check=True,
            )

            trusted = _git_safe_directories(worktree)
            clone = isolated_clone(worktree, root / "clone")

            self.assertEqual(trusted[0], worktree.resolve())
            self.assertTrue(any("worktrees" in path.parts for path in trusted[1:]))
            self.assertEqual(
                (clone / "tracked.txt").read_text(encoding="utf-8"), "content\n"
            )

    def test_dreamplace_link_is_materialized_only_for_a_source_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dependency = root / "dependency"
            source = root / "source"
            candidate = root / "candidate"
            relative = Path("dreamplace/PlaceObj.py")
            (dependency / relative).parent.mkdir(parents=True)
            (dependency / relative).write_text("BASE = 1\n", encoding="utf-8")
            (dependency / "dreamplace/unchanged.py").write_text("KEEP = 1\n", encoding="utf-8")
            for checkout in (source, candidate):
                linked = checkout / "Place-3D/thirdparty/DREAMPlace"
                linked.parent.mkdir(parents=True)
                linked.symlink_to(dependency, target_is_directory=True)

            with _sealed_fixture(dependency) as before:
                materialized = materialize_dreamplace_source_tree(
                    source,
                    candidate,
                    ["Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"],
                )
                self.assertIsNotNone(materialized)
                assert materialized is not None
                self.assertFalse(materialized.is_symlink())
                self.assertTrue(materialized.stat().st_mode & stat.S_IWUSR)
                self.assertTrue((materialized / relative).stat().st_mode & stat.S_IWUSR)
                self.assertFalse((materialized / "dreamplace/unchanged.py").stat().st_mode & stat.S_IWUSR)
                self.assertNotEqual((materialized / relative).stat().st_ino, (dependency / relative).stat().st_ino)
                (materialized / relative).write_text("BASE = 2\n", encoding="utf-8")
                self.assertEqual(_tree_identity(dependency), before)

    def test_dreamplace_materialization_rejects_mutable_source_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, candidate, dependency = root / "source", root / "candidate", root / "dependency"
            python_root = dependency / "dreamplace"
            python_root.mkdir(parents=True)
            external = root / "external"
            external.mkdir()
            (external / "PlaceObj.py").write_text("BASE = 1\n")
            (python_root / "PlaceObj.py").symlink_to(external / "PlaceObj.py")
            for checkout in (source, candidate):
                link = checkout / "Place-3D/thirdparty/DREAMPlace"
                link.parent.mkdir(parents=True)
                link.symlink_to(dependency, target_is_directory=True)
            with _sealed_fixture(external) as before:
                with self.assertRaisesRegex(WorkspaceError, "must not be a symlink"):
                    materialize_dreamplace_source_tree(source, candidate, [
                        "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"])
                self.assertEqual(_tree_identity(external), before)

    def test_candidate_patch_accepts_exact_one_sided_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            source = repo / "example.py"
            source.write_text("one\ntwo\nthree\n", encoding="utf-8")
            patch = repo / "candidate.patch"
            patch.write_text(
                "--- a/example.py\n"
                "+++ b/example.py\n"
                "@@ -1,2 +1,3 @@\n"
                " one\n"
                " two\n"
                "+inserted\n",
                encoding="utf-8",
            )

            result = apply_candidate_patch(repo, patch)

            self.assertEqual(result["mode"], "unidiff-zero")
            self.assertEqual(source.read_text(encoding="utf-8"), "one\ntwo\ninserted\nthree\n")

    def test_source_validation_rejects_invalid_python_before_eda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            source = repo / "broken.py"
            source.write_text("def broken(:\n", encoding="utf-8")
            with self.assertRaisesRegex(WorkspaceError, "does not compile"):
                validate_candidate_workspace(repo, ["broken.py"], sys.executable)

    def test_patched_dreamplace_python_reaches_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_repo = root / "source"
            install = source_repo / "Place-3D" / "build" / "dreamplace" / "install" / "dreamplace"
            install.mkdir(parents=True)
            (install / "PlaceObj.py").write_text("BASE = True\n", encoding="utf-8")
            (install / "native.so").write_bytes(b"native")
            candidate = root / "candidate"
            patched = candidate / "Place-3D" / "thirdparty" / "DREAMPlace" / "dreamplace"
            patched.mkdir(parents=True)
            (patched / "PlaceObj.py").write_text("BASE = False\n", encoding="utf-8")
            with _sealed_fixture(install.parent) as before, _sealed_fixture(patched) as patched_before:
                overlay = prepare_dreamplace_overlay(
                    source_repo,
                    candidate,
                    root / "evaluation",
                    ["Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"],
                )
                self.assertIsNotNone(overlay)
                self.assertEqual(
                    (overlay / "dreamplace" / "PlaceObj.py").read_text(encoding="utf-8"),
                    "BASE = False\n",
                )
                self.assertTrue((overlay / "dreamplace").stat().st_mode & stat.S_IWUSR)
                self.assertTrue((overlay / "dreamplace/PlaceObj.py").stat().st_mode & stat.S_IWUSR)
                self.assertEqual((overlay / "dreamplace/PlaceObj.py").stat().st_nlink, 1)
                native = overlay / "dreamplace/native.so"
                self.assertEqual(native.read_bytes(), b"native")
                self.assertEqual(native.stat().st_ino, (install / "native.so").stat().st_ino)
                self.assertFalse(native.stat().st_mode & stat.S_IWUSR)
                self.assertEqual(_tree_identity(install.parent), before)
                self.assertEqual(_tree_identity(patched), patched_before)

    def test_dreamplace_overlay_rejects_directory_symlink_before_replacing_python(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external = root / "external"
            external.mkdir()
            (external / "PlaceObj.py").write_text("BASE = True\n")
            install = root / "source/Place-3D/build/dreamplace/install"
            install.mkdir(parents=True)
            (install / "dreamplace").symlink_to(external, target_is_directory=True)
            candidate = root / "candidate"
            patched = candidate / "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"
            patched.parent.mkdir(parents=True)
            patched.write_text("BASE = False\n")
            with _sealed_fixture(external) as before:
                with self.assertRaisesRegex(WorkspaceError, "must not be a symlink"):
                    prepare_dreamplace_overlay(root / "source", candidate, root / "evaluation", [
                        "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"])
                self.assertEqual(_tree_identity(external), before)


class DryRunArtifactTest(unittest.TestCase):
    def test_dry_run_persists_the_advertised_result(self) -> None:
        spec = load_task(TASKS / "global_routing_ripup_policy")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            open3dbench = root / "Open3DBench-V2.0"
            open3dbench.mkdir()
            # This dry-run fixture represents the supported standalone GRT API.
            backend = open3dbench / "open3dbench/backend"
            backend.mkdir(parents=True)
            (backend.parent / "runner.py").write_text(
                "def global_route():\n    pass\n# grt_only=True, output_dir=destination\n")
            (backend / "three_d_route.py").write_text(
                '\"global_route_checkpoint.json\"\n\"grt_metrics.json\"\n')
            output = root / "evaluation"
            completed = mock.Mock(returncode=0, stdout='{"plan": []}\n')
            with mock.patch(
                "local_bench.evaluation.subprocess.run", return_value=completed
            ) as run_mock:
                result = evaluate_task(
                    spec.root,
                    spec.initial_candidate,
                    open3dbench,
                    output,
                    dry_run=True,
                )
            plan_environment = run_mock.call_args.kwargs["env"]
            self.assertEqual(
                plan_environment["PYTHONPATH"].split(os.pathsep)[0],
                str(open3dbench.resolve()),
            )
            result_path = Path(result.artifacts["evaluation_result"])
            self.assertTrue(result_path.is_file())
            persisted = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["metrics"]["dry_run"], 1.0)
            self.assertEqual(persisted["artifacts"]["cascade"][-1]["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
