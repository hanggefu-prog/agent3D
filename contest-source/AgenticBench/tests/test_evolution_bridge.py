from __future__ import annotations

import ast
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from local_bench.evolve_protocol import parse_candidate_program, render_candidate_program
from local_bench.llm_client import OpenAICompatibleLLM, load_llm_config
from local_bench.llm_client import ReplayLLM
from local_bench.evostage_adapter import (
    EvolutionOptions,
    _seed_frozen_reference,
    run_evolution,
)
from local_bench.model import load_candidate, load_task
from local_bench.model import SourceRange
from local_bench.openevolve_adapter import (
    _docker_evaluation_command,
    _evaluate_via_docker,
    _gpu_evaluation_lock,
    _materialize_candidate,
)
from local_bench.program_bridge import _load_trusted_adapter, initial_program_text, materialize_program
from local_bench.prompting import (
    ContextualLLM,
    build_problem_description,
    composed_system_message,
    export_openevolve_bundle,
)
from local_bench.verifier import verify_candidate


ROOT = Path(__file__).resolve().parents[1]


def literal_spec():
    return replace(load_task(ROOT / "local_tasks/placement_initialization"),
                   program_mode="candidate_literal", program_template=None,
                   program_adapter=None, program_entrypoint="evolve_candidate")


def _write_fake_grt_runtime(root: Path) -> None:
    route = root / "open3dbench/backend/three_d_route.py"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text(
        "import json\n"
        "\n"
        "ROUTE_TCL = r\"\"\"\n"
        "write_guides $::env(ROUTE_GUIDE)\n"
        "\n"
        "set drt_args\n"
        "\"\"\"\n"
        "\n"
        "def _wire_length_summary(log: Path) -> Dict[str, object]:\n"
        "    return {}\n"
        "\n"
        "def _run_openroad(\n"
        "    script: Path,\n"
        "    log: Path,\n"
        "    executable: str,\n"
        "    *,\n"
        "    resource_reduction: float = 0.0,\n"
        ") -> Dict[str, object]:\n"
        "    with log.open(\"w\", encoding=\"utf-8\") as output:\n"
        "        process = subprocess.run(\n"
        "            [executable],\n"
        "            env=dict(os.environ),\n"
        "        )\n"
        "\n"
        "def route_two_dies():\n"
        "    started = time.perf_counter()\n"
        "    checkpoint_path = checkpoint_path.resolve()\n"
        "            warning_audit = _run_openroad(\n"
        "                script,\n"
        "                die_dir / \"route.log\",\n"
        "                openroad,\n"
        "                resource_reduction=resource_reduction,\n"
        "            )\n"
        "        boundary_rows = [\n"
        "        ]\n"
        "    _copy_continuation(source, output_dir)\n"
        "    total_drc = sum(int(dies[die][\"drc_violations\"]) for die in dies)\n",
        encoding="utf-8",
    )
    runner = root / "open3dbench/runner.py"
    runner.write_text(
        "class Runner:\n"
        "    def _run_routing(self):\n"
        "        artifacts = {name: root / relative for name, relative in names.items()}\n"
        "        self._command(command, self.repo, stage / \"routing.log\")\n"
        "        validate_checkpoint(stage / \"checkpoint.json\", require_files=True)\n"
        "\n"
        "    def _run_report(self) -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )


class LiteralCandidateProtocolTest(unittest.TestCase):
    def test_task_adapter_openevolve_configs_request_complete_programs(self) -> None:
        for task_name in ("placement_parameters_reference_20260923", "placement_preconditioner"):
            config = yaml.safe_load(
                (ROOT / "local_tasks" / task_name / "config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(config["diff_based_evolution"])

    def test_placement_tasks_share_the_same_search_database_budget(self) -> None:
        databases = []
        for task_name in ("placement_parameters_reference_20260923", "placement_preconditioner"):
            config = yaml.safe_load(
                (ROOT / "local_tasks" / task_name / "config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            databases.append(config["database"])
        self.assertEqual(databases[0], databases[1])
        self.assertEqual(
            databases[0],
            {"population_size": 10, "archive_size": 8, "num_islands": 2},
        )

    def test_source_prompt_states_initializer_call_grammar(self):
        prompt = (ROOT / "local_tasks/placement_initialization/prompt.md").read_text()
        for word in ("initialize_position", "die_contexts", "hasattr", "local numerical helpers"):
            self.assertIn(word, prompt)

    def test_placement_prompts_document_real_data_without_case_sizes(self):
        for task in ("placement_initialization", "placement_preconditioner", "placement_parameters_reference_20260923"):
            prompt = (ROOT / "local_tasks" / task / "prompt.md").read_text()
            self.assertIn("Input reference", prompt)
            self.assertNotIn("length: 371462", prompt)

    def test_openevolve_docker_bridge_builds_bounded_cli_command(self) -> None:
        environment = {
            "OPENAGENTIC3D_EVAL_DOCKER_IMAGE": "runtime:test",
            "OPENAGENTIC3D_EVAL_DOCKER_GPU": "2",
            "OPENAGENTIC3D_EVAL_DOCKER_USER": "1005:1005",
            "OPENAGENTIC3D_EVAL_DOCKER_MOUNTS_JSON": json.dumps(
                ["/data/project:/data/project", "/runtime:/runtime:ro"]
            ),
            "OPENAGENTIC3D_EVAL_DOCKER_ENV_JSON": json.dumps(
                {"PYTHONPATH": "/framework:/dreamplace",
                 "CCACHE_DIR": "/data/run/openroad-ccache",
                 "CCACHE_MAXSIZE": "8G"}
            ),
            "OPENAGENTIC3D_EVAL_DOCKER_WORKDIR": "/framework",
            "OPENAGENTIC3D_BENCHMARK_INPUT": "/data/input",
            "OPEN3DBENCH_REFERENCE_RUN": "/data/reference",
            "OPENAGENTIC3D_BASE_CONFIG": "/data/case.yaml",
            "OPEN3D_LOCAL_EVAL_TIMEOUT": "900",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            command = _docker_evaluation_command(
                Path("/framework/local_tasks/task"),
                Path("/data/candidate.py"),
                Path("/data/Open3DBench-V2.0"),
                Path("/data/evaluation"),
            )
        self.assertEqual(command[:5], ["docker", "run", "--rm", "--user", "1005:1005"])
        self.assertIn("--gpus", command)
        self.assertIn("device=2", command)
        self.assertIn("runtime:test", command)
        self.assertIn("/data/input", command)
        self.assertIn("/data/reference", command)
        self.assertIn("/data/case.yaml", command)
        self.assertIn("CCACHE_DIR=/data/run/openroad-ccache", command)
        self.assertIn("CCACHE_MAXSIZE=8G", command)
        self.assertIn("--discard-workspace", command)

    def test_openevolve_docker_bridge_materializes_host_temp_program(self) -> None:
        task_dir = ROOT / "local_tasks" / "placement_initialization"
        spec = load_task(task_dir)
        program_text = initial_program_text(spec, load_candidate(spec.initial_candidate))
        environment = {
            "OPENAGENTIC3D_EVAL_DOCKER_IMAGE": "runtime:test",
            "OPENAGENTIC3D_EVAL_DOCKER_MOUNTS_JSON": "[]",
            "OPENAGENTIC3D_EVAL_DOCKER_ENV_JSON": "{}",
            "OPEN3D_LOCAL_EVAL_TIMEOUT": "30",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "candidate-from-openevolve.py"
            program.write_text(program_text, encoding="utf-8")
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                "local_bench.openevolve_adapter.subprocess.run",
                return_value=SimpleNamespace(returncode=7, stdout="expected test failure"),
            ) as run:
                _evaluate_via_docker(task_dir, program, ROOT.parent, root / "evaluations")
            command = run.call_args.args[0]
            candidate_arg = Path(command[command.index("--candidate") + 1])
            self.assertTrue(candidate_arg.is_dir())
            self.assertTrue((candidate_arg / "candidate.json").is_file())
            self.assertNotEqual(candidate_arg, program)

    def test_openevolve_docker_bridge_can_compact_reproducible_artifacts(self) -> None:
        environment = {
            "OPENAGENTIC3D_EVAL_DOCKER_IMAGE": "runtime:test",
            "OPENAGENTIC3D_EVAL_COMPACT_ARTIFACTS": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            command = _docker_evaluation_command(
                Path("/framework/local_tasks/task"),
                Path("/data/candidate.py"),
                Path("/data/Open3DBench-V2.0"),
                Path("/data/evaluation"),
            )
        self.assertIn("--compact-artifacts", command)
        self.assertNotIn("--discard-workspace", command)

    def test_gpu_evaluation_lock_is_opt_in_and_gpu_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = {
                "OPENAGENTIC3D_EVAL_GPU_LOCK_DIR": tmp,
                "OPENAGENTIC3D_EVAL_DOCKER_GPU": "2",
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                "local_bench.openevolve_adapter.fcntl.flock"
            ) as flock:
                with _gpu_evaluation_lock():
                    self.assertTrue((Path(tmp) / "gpu-2.lock").is_file())
            self.assertEqual(flock.call_count, 2)

    def test_round_trip_initial_candidates(self) -> None:
        for task_dir in sorted((ROOT / "local_tasks").iterdir()):
            if not (task_dir / "task.yaml").is_file():
                continue
            spec = load_task(task_dir)
            if spec.program_mode != "candidate_literal":
                continue
            candidate = load_candidate(spec.initial_candidate)
            decoded = parse_candidate_program(render_candidate_program(candidate))
            self.assertEqual(decoded, candidate)
            verify_candidate(spec, decoded)

    def test_native_openevolve_initial_programs_match_candidates(self) -> None:
        for task_dir in sorted((ROOT / "local_tasks").iterdir()):
            if not (task_dir / "task.yaml").is_file():
                continue
            spec = load_task(task_dir)
            program = initial_program_text(spec, load_candidate(spec.initial_candidate))
            self.assertIn("# EVOLVE-BLOCK-START", program)
            self.assertIn("# EVOLVE-BLOCK-END", program)
            if spec.program_mode == "candidate_literal":
                expected = load_candidate(spec.initial_candidate)
                self.assertEqual(parse_candidate_program(program), expected)
            else:
                self.assertIn(spec.program_entrypoint, program)

    def test_current_program_adapters_materialize_bounded_patches(self):
        expected = {
            "placement_initialization": {"Place-3D/thirdparty/DREAMPlace/dreamplace/BasicPlace.py"},
            "placement_preconditioner": {"Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"},
            "placement_parameters_reference_20260923": {
                "Place-3D/thirdparty/DREAMPlace/dreamplace/NonLinearPlace.py",
                "Place-3D/thirdparty/DREAMPlace/dreamplace/NesterovAcceleratedGradientOptimizer.py"},
            "tp_gnn_post_placement_timing": {"Place-3D/src/partition/lol/methods/tp_gnn.py"},
        }
        for name, paths in expected.items():
            with self.subTest(task=name):
                spec = load_task(ROOT / "local_tasks" / name)
                program = initial_program_text(spec, load_candidate(spec.initial_candidate))
                program = program.replace("* 0.001", "* 0.002").replace("self.alpha = 1.0", "self.alpha = 1.1")
                candidate = materialize_program(spec, program, ROOT.parent)
                self.assertEqual(set(verify_candidate(spec, candidate)["patch"]["paths"]), paths)
                self.assertNotIn('+        "optimizer": "adam",', candidate["source_patch"])

    def test_tp_gnn_objective_retains_native_autograd_guard(self):
        from local_tasks import tp_gnn_timing_context as adapter
        spec = load_task(ROOT / "local_tasks/tp_gnn_post_placement_timing")
        unsafe = "def compute_tp_gnn_loss(" + ", ".join(adapter.OBJECTIVE_ARGS) + "):\n    return positive_scores.detach().mean()\n"
        candidate = materialize_program(spec, unsafe, ROOT.parent)
        self.assertIn("if not loss.requires_grad", candidate["source_patch"])
        self.assertIn("detached the training objective", candidate["source_patch"])

    def test_tp_gnn_objective_accepts_differentiable_tanh(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "tp_gnn_post_placement_timing")
        from local_tasks import tp_gnn_timing_context as adapter
        signature = "def compute_tp_gnn_loss(" + ", ".join(adapter.OBJECTIVE_ARGS) + "):"
        program = signature + "\n    return torch.tanh(positive_scores).mean()\n"
        adapter._validate_objective(ast.parse(program).body[0], adapter.OBJECTIVE_ARGS)
        prompt = (ROOT / "local_tasks/tp_gnn_post_placement_timing/prompt.md").read_text()
        self.assertIn("`tanh`", prompt)

    def test_tp_gnn_tanh_support_does_not_admit_side_effects(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "tp_gnn_post_placement_timing")
        from local_tasks import tp_gnn_timing_context as adapter
        signature = "def compute_tp_gnn_loss(" + ", ".join(adapter.OBJECTIVE_ARGS) + "):"
        for expression in ("torch.load('input.pt')", "torch.save(positive_scores, 'output.pt')",
                           "torch.rand(4)", "positive_scores.tanh_()", "positive_scores.requires_grad_(False)"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                adapter._validate_objective(ast.parse(signature + "\n    return " + expression + "\n").body[0], adapter.OBJECTIVE_ARGS)

    def test_dynamic_program_adapter_rejects_external_effects(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_parameters_reference_20260923")
        unsafe = """import os
def adjust_learning_rate(native_step_size, step_num, log_hpwl, log_hpwl_prev, overflow_by_die, log_lambda_by_die, step_multiplier_prev, log_gradient_norm_by_die, die_multipliers_prev):
    return step_multiplier_prev, (1.0, 1.0)
"""
        with self.assertRaisesRegex(ValueError, "exactly one FunctionDef"):
            materialize_program(spec, unsafe, Path("/does/not/matter"))



    def test_active_routing_policy_adapters_materialize_real_hot_path_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_grt_runtime(root)
            source_root = (
                root / "OpenROAD-3D/openroad/src/grt/src/fastroute/src"
            )
            source_root.mkdir(parents=True)
            (source_root / "RipUp.cpp").write_text(
                "#include <cstdint>\n"
                "#include <limits>\n"
                "#include \"FastRoute.h\"\n"
                "namespace grt {\n"
                "using utl::GRT;\n"
                "void fakeRipup() {\n"
                "  if (needRipup) {\n"
                "    const int8_t edgeCost = net->getEdgeCost();\n"
                "  }\n"
                "}\n"
                "}\n",
                encoding="utf-8",
            )
            (source_root / "maze.cpp").write_text(
                "#include <cstdint>\n"
                "#include <unordered_map>\n"
                "#include \"Graph2D.h\"\n"
                "namespace grt {\n"
                "using utl::GRT;\n"
                "using utl::Timer;\n"
                "void fakeMaze() {\n"
                "  auto relaxAdjacent = [&](const int cur_x,\n"
                "                            const int cur_y) {\n"
                "    double cost1 = getCost(pos1, is_horizontal, cost_params);\n"
                "        double cost2 = getCost(pos2, is_horizontal, cost_params);\n"
                "  };\n"
                "}\n"
                "}\n",
                encoding="utf-8",
            )
            (source_root / "FastRoute.cpp").write_text(
                "#include <cstdint>\n"
                "namespace grt {\n"
                "using utl::GRT;\n"
                "void fakeIteration() {\n"
                "      const int snapshot_batch_count_before = snapshot_batch_count_;\n"
                "}\n"
                "}\n",
                encoding="utf-8",
            )

            cases = (
                (
                    "global_routing_ripup_policy",
                    "OpenROAD-3D/openroad/src/grt/src/fastroute/src/RipUp.cpp",
                    "Open3DRipupContext",
                    "OPEN3D_RIPUP_POLICY die={}",
                ),
                (
                    "global_routing_maze_cost",
                    "OpenROAD-3D/openroad/src/grt/src/fastroute/src/maze.cpp",
                    "Open3DMazeStepContext",
                    "OPEN3D_MAZE_COST die={}",
                ),
                (
                    "global_routing_iteration_controller",
                    "OpenROAD-3D/openroad/src/grt/src/fastroute/src/FastRoute.cpp",
                    "Open3DIterationContext",
                    "OPEN3D_ITERATION_CONTROL die={}",
                ),
            )
            for task, source, context, diagnostic in cases:
                spec = load_task(ROOT / "local_tasks" / task)
                program = initial_program_text(
                    spec, load_candidate(spec.initial_candidate)
                )
                candidate = materialize_program(spec, program, root)
                # Routing adapters can include trusted evaluator wrappers. Their
                # byte identity must be checked against this exact source fixture,
                # rather than silently falling back to scope-only verification.
                with self.assertRaisesRegex(ValueError, "require the Open3DBench checkout"):
                    verify_candidate(spec, candidate)
                report = verify_candidate(spec, candidate, root)
                self.assertEqual(
                    set(report["patch"]["paths"]),
                    {
                        source,
                        "open3dbench/backend/three_d_route.py",
                        "open3dbench/runner.py",
                    },
                )
                patch = candidate["source_patch"]
                self.assertIn(context, patch)
                self.assertIn(diagnostic, patch)
                self.assertIn("OPEN3D_ROUTING_DIE", patch)
                self.assertIn("OPEN3D_HBT_PORT_FILE", patch)
                self.assertIn("OPEN3D_GRT_ONLY", patch)

            ripup = load_task(ROOT / "local_tasks" / "global_routing_ripup_policy")
            candidate = materialize_program(
                ripup,
                "def score_ripup_edge(ctx):\n"
                "    score = ctx.baseline_decision\n"
                "    if ctx.die_id > 0.5:\n"
                "        score += 0.2 * ctx.edge_overflow\n"
                "    else:\n"
                "        score -= 0.1 * ctx.detour_ratio\n"
                "    for step in range(4):\n"
                "        score += 0.01 * step * ctx.history_pressure\n"
                "    return score\n",
                root,
            )
            self.assertIn(
                "for (int step = 0; step < 4; step += 1)",
                candidate["source_patch"],
            )

            maze = load_task(ROOT / "local_tasks" / "global_routing_maze_cost")
            with self.assertRaisesRegex(ValueError, "loop has 9 iterations"):
                materialize_program(
                    maze,
                    "def score_maze_step(ctx):\n"
                    "    score = 1.0\n"
                    "    for step in range(9):\n"
                    "        score += 0.01 * step\n"
                    "    return score\n",
                    root,
                )

            controller = load_task(
                ROOT / "local_tasks" / "global_routing_iteration_controller"
            )
            controlled = materialize_program(
                controller,
                "def control_reroute_iteration(ctx):\n"
                "    if ctx.die_id > 0.5:\n"
                "        congestion = 1.4\n"
                "        search = 0.9\n"
                "    else:\n"
                "        congestion = 0.8\n"
                "        search = 1.2\n"
                "    offset = 2.0 * (ctx.stagnation - 0.5)\n"
                "    return congestion, search, offset\n",
                root,
            )
            self.assertIn(
                "return Open3DIterationDecision{congestion, search, offset};",
                controlled["source_patch"],
            )
            with self.assertRaisesRegex(
                ValueError, "must return exactly 3 numeric values"
            ):
                materialize_program(
                    controller,
                    "def control_reroute_iteration(ctx):\n"
                    "    return ctx.baseline_congestion_scale\n",
                    root,
                )

    def test_dynamic_program_adapter_strips_generated_trailing_whitespace(self):
        spec = load_task(ROOT / "local_tasks/placement_parameters_reference_20260923")
        program = initial_program_text(spec, load_candidate(spec.initial_candidate))
        program = "\n".join(line + "  " if line else "" for line in program.splitlines()) + "\n"
        candidate = materialize_program(spec, program, ROOT.parent)
        self.assertFalse(any(line.startswith("+") and line.rstrip() != line for line in candidate["source_patch"].splitlines()))

    def test_dynamic_controller_allows_only_statically_bounded_loops(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_parameters_reference_20260923")
        adapter = _load_trusted_adapter(spec)
        safe = ast.parse(
            "def adjust_learning_rate(native_step_size, step_num, log_hpwl, "
            "log_hpwl_prev, overflow_by_die, log_lambda_by_die, "
            "step_multiplier_prev, log_gradient_norm_by_die, "
            "die_multipliers_prev):\n"
            "    total = 0.0\n"
            "    for index in range(2):\n"
            "        total += overflow_by_die[index]\n"
            "    return max(native_step_size * 1e-4, step_multiplier_prev), (1.0, 1.0)\n"
        ).body[0]
        adapter._validate_controller(safe)
        unsafe = ast.parse(
            "def adjust_learning_rate(native_step_size, step_num, log_hpwl, "
            "log_hpwl_prev, overflow_by_die, log_lambda_by_die, "
            "step_multiplier_prev, log_gradient_norm_by_die, "
            "die_multipliers_prev):\n"
            "    for index in range(step_num):\n"
            "        pass\n"
            "    return native_step_size, (1.0, 1.0)\n"
        ).body[0]
        with self.assertRaisesRegex(ValueError, "bounded two-die"):
            adapter._validate_controller(unsafe)




    def test_source_program_adapter_rejects_torch_file_io(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_preconditioner")
        initial = initial_program_text(spec, load_candidate(spec.initial_candidate))
        unsafe = initial.replace(
            "        self.placedb = placedb",
            "        torch.save(placedb, '/tmp/leak.pt')\n        self.placedb = placedb",
        )
        with self.assertRaisesRegex(ValueError, "forbidden torch external-effect API save"):
            materialize_program(spec, unsafe, Path("/does/not/matter"))

    def test_source_program_adapter_accepts_deterministic_numpy_reductions(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_preconditioner")
        from local_tasks import placement_source_common as adapter
        function = ast.parse(
            "def initialize_position(params, placedb, die_contexts=None):\n"
            "    values = np.asarray([3.0, 1.0, 3.0])\n"
            "    present = np.any(values > 0.0)\n"
            "    index = np.argmax(values)\n"
            "    distinct = np.unique(values)\n"
            "    deltas = np.diff(np.random.permutation(values))\n"
            "    return deltas if present and index >= 0 and len(distinct) else values\n"
        ).body[0]
        adapter._validate_initializer(function)

    def test_source_program_adapter_rejects_numpy_file_io(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_preconditioner")
        from local_tasks import placement_source_common as adapter
        function = ast.parse(
            "def initialize_position(params, placedb, die_contexts=None):\n"
            "    values = np.asarray([1.0])\n"
            "    np.save('/tmp/leak.npy', values)\n"
            "    return values\n"
        ).body[0]
        with self.assertRaisesRegex(
            ValueError, "forbidden np external-effect API save"
        ):
            adapter._validate_initializer(function)

    def test_source_program_adapter_accepts_ordinary_torch_construction(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_preconditioner")
        adapter = _load_trusted_adapter(spec)
        initial = initial_program_text(spec, load_candidate(spec.initial_candidate))
        safe = initial.replace(
            "        self.placedb = placedb",
            "        self._scratch = torch.full((1,), 0.0)\n        self.placedb = placedb",
        )
        from local_tasks.placement_source_common import _validate_class
        class_node = next(node for node in ast.parse(safe).body if isinstance(node, ast.ClassDef))
        _validate_class(class_node)

    def test_rejects_generated_code_execution(self) -> None:
        program = """def evolve_candidate() -> dict:
    return __import__('os').environ
"""
        with self.assertRaisesRegex(ValueError, "static Python literal"):
            parse_candidate_program(program)

    def test_rejects_top_level_import(self) -> None:
        program = """import os
def evolve_candidate() -> dict:
    return {}
"""
        with self.assertRaisesRegex(ValueError, "exactly one candidate function"):
            parse_candidate_program(program)

    def test_rejects_unknown_candidate_field(self) -> None:
        spec = literal_spec()
        candidate = load_candidate(spec.initial_candidate)
        candidate["command"] = "unsafe"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            verify_candidate(spec, candidate)

    def test_source_patch_survives_literal_bridge(self):
        spec = load_task(ROOT / "local_tasks/placement_preconditioner")
        program = initial_program_text(spec, load_candidate(spec.initial_candidate)).replace("self.alpha = 1.0", "self.alpha = 1.1")
        candidate = materialize_program(spec, program, ROOT.parent)
        decoded = parse_candidate_program(render_candidate_program(candidate))
        self.assertEqual(candidate, decoded)
        self.assertEqual(verify_candidate(spec, decoded)["patch"]["paths"],
                         ["Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"])


class LLMConfigurationTest(unittest.TestCase):
    def test_key_is_environment_only_and_not_in_public_report(self) -> None:
        config = load_llm_config(ROOT / "local_bench" / "configs" / "qwen.yaml")
        with mock.patch.dict(os.environ, {config.api_key_env: "secret-test-value"}, clear=False):
            report = config.public_dict()
            client = OpenAICompatibleLLM(config)
            self.assertTrue(report["api_key_configured"])
            self.assertNotIn("secret-test-value", json.dumps(report))
            self.assertNotIn("secret-test-value", repr(vars(client)))
            client.close()

    def test_deepseek_stream_collects_content_and_ignores_reasoning(self) -> None:
        config = load_llm_config(ROOT / "local_bench" / "configs" / "deepseek.yaml")
        with mock.patch.dict(os.environ, {config.api_key_env: "secret-test-value"}, clear=False):
            client = OpenAICompatibleLLM(config)
            text = client._stream_chat_text(
                [
                    b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"first"}}]}\n',
                    b': keep-alive\n',
                    b'data: {"choices":[{"delta":{"content":" second"}}]}\n',
                    b'data: [DONE]\n',
                ]
            )
            self.assertTrue(config.stream)
            self.assertEqual(config.model, "deepseek-v4-flash")
            self.assertEqual(config.reasoning_effort, "high")
            self.assertEqual(config.max_tokens, 8192)
            self.assertEqual(text, "first second")
            client.close()

    def test_deepseek_thinking_mode_is_explicit_in_request(self) -> None:
        config = load_llm_config(ROOT / "local_bench" / "configs" / "deepseek.yaml")

        class StreamingResponse:
            def __enter__(self):
                return iter(
                    [
                        b'data: {"choices":[{"delta":{"content":"candidate"}}]}\n',
                        b'data: [DONE]\n',
                    ]
                )

            def __exit__(self, exc_type, exc, traceback):
                return False

        with mock.patch.dict(os.environ, {config.api_key_env: "secret-test-value"}, clear=False):
            with mock.patch(
                "local_bench.llm_client.urllib.request.urlopen",
                return_value=StreamingResponse(),
            ) as urlopen:
                client = OpenAICompatibleLLM(config)
                self.assertEqual(client.draw_sample("generate"), "candidate")
                request = urlopen.call_args.args[0]
                body = json.loads(request.data.decode("utf-8"))
                self.assertEqual(
                    body["thinking"],
                    {"type": "enabled"},
                )
                self.assertEqual(body["reasoning_effort"], "high")
                self.assertNotIn("temperature", body)
                self.assertNotIn("top_p", body)
                self.assertEqual(body["max_tokens"], 8192)
                client.close()

    def test_deepseek_empty_thinking_stream_retries_without_thinking(self) -> None:
        config = load_llm_config(ROOT / "local_bench" / "configs" / "deepseek.yaml")

        class StreamingResponse:
            def __init__(self, rows):
                self.rows = rows

            def __enter__(self):
                return iter(self.rows)

            def __exit__(self, exc_type, exc, traceback):
                return False

        responses = [
            StreamingResponse(
                [
                    b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n',
                    b'data: [DONE]\n',
                ]
            ),
            StreamingResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"candidate"}}]}\n',
                    b'data: [DONE]\n',
                ]
            ),
        ]
        with mock.patch.dict(
            os.environ, {config.api_key_env: "secret-test-value"}, clear=False
        ), mock.patch(
            "local_bench.llm_client.urllib.request.urlopen",
            side_effect=responses,
        ) as urlopen, mock.patch(
            "local_bench.llm_client.time.sleep"
        ):
            client = OpenAICompatibleLLM(config)
            self.assertEqual(client.draw_sample("generate"), "candidate")
            first = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))
            second = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
            self.assertEqual(
                first["thinking"],
                {"type": "enabled"},
            )
            self.assertEqual(first["reasoning_effort"], "high")
            self.assertEqual(second["thinking"], {"type": "disabled"})
            self.assertNotIn("reasoning_effort", second)
            client.close()


class PromptAndFeedbackContractTest(unittest.TestCase):
    def test_llm_repairs_preflight_rejection_before_spending_sample(self) -> None:
        spec = literal_spec()
        valid = render_candidate_program(load_candidate(spec.initial_candidate))

        class RepairProvider:
            def __init__(self):
                self.calls = []

            def draw_sample(self, messages, *args, **kwargs):
                self.calls.append(messages)
                return "not a candidate program" if len(self.calls) == 1 else valid

        provider = RepairProvider()
        harness = SimpleNamespace(
            spec=spec,
            records=[],
            options=SimpleNamespace(open3dbench_root=None),
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            client = ContextualLLM(provider, harness, trace)
            response = client.draw_sample("generate")
            record = json.loads(trace.read_text(encoding="utf-8"))
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("return", response)
        self.assertIsNotNone(record["provider_attempts"][0]["preflight_error"])
        self.assertIsNone(record["provider_attempts"][1]["preflight_error"])
        self.assertIn(
            "trusted local verifier rejected",
            provider.calls[1][-1]["content"],
        )

    def test_routing_reference_policy_is_seeded_at_normalized_score_one(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "global_routing_ripup_policy")

        class Population:
            def __init__(self):
                self.items = []

            def register_function(self, function):
                self.items.append(function)

        population = Population()
        method = SimpleNamespace(
            _function_to_evolve_list=[
                SimpleNamespace(
                    name="score_ripup_edge",
                    body="    return ctx.baseline_decision",
                    score=None,
                )
            ],
            _population_list=[population],
        )
        harness = SimpleNamespace(
            spec=spec,
            options=SimpleNamespace(
                evaluation_mode="full",
                reference_run=Path("/frozen/reference"),
            ),
        )
        self.assertTrue(_seed_frozen_reference(method, harness))
        self.assertEqual(len(population.items), 1)
        self.assertEqual(population.items[0].score, 1.0)
        self.assertIn("Frozen maintained", population.items[0].algorithm)

    def test_task_specific_system_prompt_is_used_by_all_search_backends(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_parameters_reference_20260923")
        expected = "Nesterov gradient-step controller"
        self.assertIn(expected, composed_system_message(spec))

        class CaptureProvider:
            def __init__(self):
                self.messages = None

            def draw_sample(self, messages, *args, **kwargs):
                self.messages = messages
                return initial_program_text(spec, load_candidate(spec.initial_candidate))

        provider = CaptureProvider()
        harness = SimpleNamespace(spec=spec, records=[])
        with tempfile.TemporaryDirectory() as tmp:
            client = ContextualLLM(provider, harness, Path(tmp) / "trace.jsonl")
            client.draw_sample("generate")
            self.assertIn(expected, provider.messages[0]["content"])

            with mock.patch(
                "local_bench.prompting.build_problem_description",
                return_value="resolved benchmark description",
            ):
                bundle = export_openevolve_bundle(
                    spec,
                    load_candidate(spec.initial_candidate),
                    Path(tmp),
                    Path(tmp) / "bundle",
                )
            config = yaml.safe_load(Path(bundle["config"]).read_text(encoding="utf-8"))
            self.assertIn(expected, config["prompt"]["system_message"])

    def test_joint_source_program_uses_static_string_transport(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_preconditioner")

        class Provider:
            def draw_sample(self, messages, *args, **kwargs):
                return initial_program_text(spec, load_candidate(spec.initial_candidate))

        harness = SimpleNamespace(spec=spec, records=[])
        with tempfile.TemporaryDirectory() as tmp:
            client = ContextualLLM(Provider(), harness, Path(tmp) / "trace.jsonl")
            transported = client.draw_sample("generate")
        self.assertTrue(transported.startswith("    return "))
        self.assertNotIn("initialize_position", transported)
        self.assertIn("PreconditionOp", transported)

    def test_problem_description_contains_policy_and_exact_source_excerpt(self) -> None:
        spec = load_task(ROOT / "local_tasks" / "placement_preconditioner")
        with tempfile.TemporaryDirectory() as tmp:
            open3dbench = Path(tmp)
            source = open3dbench / "algo.py"
            source.write_text("def seed():\n    return 1\n", encoding="utf-8")
            prompt_spec = replace(
                spec,
                allowed_source_paths=["algo.py"],
                prompt_source_ranges=[SourceRange("algo.py", 1, 2)],
            )
            description = build_problem_description(
                prompt_spec,
                load_candidate(spec.initial_candidate),
                open3dbench,
            )
        self.assertIn("Initial program and exact interface", description)
        self.assertIn("class PreconditionOp:", description)
        self.assertNotIn('"allowed_source_paths": [', description)
        self.assertNotIn("def seed():", description)

    def test_llm_receives_metrics_artifacts_and_recent_history(self) -> None:
        spec = literal_spec()

        class CaptureProvider:
            def __init__(self):
                self.messages = None

            def draw_sample(self, messages, *args, **kwargs):
                self.messages = messages
                return "response"

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stage.log"
            log.write_text("routing overflow remains on top die\n", encoding="utf-8")
            harness = SimpleNamespace(
                spec=spec,
                records=[{
                    "order": 1,
                    "status": "accepted",
                    "score": 0.75,
                    "metrics": {"combined_score": 0.75, "placement_native_hpwl": 12},
                    "artifacts": {"stage_log": str(log), "cascade": []},
                }],
            )
            provider = CaptureProvider()
            llm = ContextualLLM(provider, harness, Path(tmp) / "trace.jsonl")
            self.assertEqual(llm.draw_sample("improve candidate"), "response")
            rendered = json.dumps(provider.messages)
            self.assertIn("placement_native_hpwl", rendered)
            self.assertNotIn("routing overflow remains on top die", rendered)
            self.assertIn("recent_attempts", rendered)

    def test_openevolve_program_is_materialized_instead_of_ignored(self):
        task_dir = ROOT / "local_tasks/placement_preconditioner"
        spec = load_task(task_dir)
        program_text = initial_program_text(spec, load_candidate(spec.initial_candidate)).replace("self.alpha = 1.0", "self.alpha = 1.1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "program.py"
            program.write_text(program_text)
            materialized = _materialize_candidate(task_dir, program, root / "evaluation", ROOT.parent)
            decoded = json.loads((materialized / "candidate.json").read_text())
        self.assertIn("+        self.alpha = 1.1", decoded["source_patch"])

    def test_native_openevolve_export_resolves_complete_contract(self) -> None:
        import yaml

        task_dir = ROOT / "local_tasks" / "placement_initialization"
        spec = literal_spec()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = export_openevolve_bundle(
                spec,
                load_candidate(spec.initial_candidate),
                root,
                root / "bundle",
            )
            config = yaml.safe_load(Path(bundle["config"]).read_text(encoding="utf-8"))
            program = parse_candidate_program(
                Path(bundle["initial_program"]).read_text(encoding="utf-8")
            )
            evaluator = Path(bundle["evaluator"]).read_text(encoding="utf-8")
        self.assertEqual(program, load_candidate(spec.initial_candidate))
        self.assertTrue(config["prompt"]["include_artifacts"])
        self.assertTrue(config["evaluator"]["enable_artifacts"])
        self.assertFalse(config["evaluator"]["cascade_evaluation"])
        self.assertIn("initialize_position", config["prompt"]["system_message"])
        self.assertIn("evaluate_from_task_dir", evaluator)
        self.assertIn("BASE_CONFIG", evaluator)

    def test_openai_responses_text_is_collected_without_reasoning_items(self) -> None:
        config = load_llm_config(ROOT / "local_bench" / "configs" / "openai.yaml")
        with mock.patch.dict(os.environ, {config.api_key_env: "secret-test-value"}, clear=False):
            client = OpenAICompatibleLLM(config)
            text = client._response_text({
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "first"},
                            {"type": "refusal", "refusal": "ignored"},
                            {"type": "output_text", "text": "second"},
                        ],
                    },
                ]
            })
            self.assertEqual(text, "first\nsecond")
            self.assertEqual(config.api_mode, "responses")
            self.assertEqual(config.api_key_env, "OPENAI_API_KEY")
            client.close()

    def test_openai_compatible_template_builds_chat_request(self) -> None:
        config = load_llm_config(
            ROOT
            / "local_bench"
            / "configs"
            / "openai_compatible.example.yaml"
        )
        response = io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": "OK"}}]}
            ).encode("utf-8")
        )
        with mock.patch.dict(
            os.environ, {config.api_key_env: "secret-test-value"}, clear=False
        ), mock.patch(
            "local_bench.llm_client.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            client = OpenAICompatibleLLM(config)
            self.assertEqual(client.draw_sample("Reply with exactly: OK"), "OK")
            request = urlopen.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(
                request.full_url,
                "https://YOUR_OPENAI_COMPATIBLE_HOST/v1/chat/completions",
            )
            self.assertEqual(payload["model"], "YOUR_MODEL_ID")
            self.assertEqual(
                payload["messages"],
                [{"role": "user", "content": "Reply with exactly: OK"}],
            )
            self.assertNotIn("reasoning", payload)
            self.assertNotIn("thinking", payload)
            self.assertNotIn("secret-test-value", request.data.decode("utf-8"))
            client.close()


class ExternalEvoStageIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("EVOSTAGE_ROOT"), "set EVOSTAGE_ROOT for external integration")
    def test_real_randsample_class_uses_literal_bridge(self) -> None:
        task_dir = ROOT / "local_tasks" / "placement_initialization"
        spec = load_task(task_dir)
        provider = ReplayLLM(load_candidate(spec.initial_candidate))
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_evolution(
                "randsample",
                provider,
                EvolutionOptions(
                    task_dir=task_dir,
                    evostage_root=Path(os.environ["EVOSTAGE_ROOT"]),
                    output_dir=Path(tmp),
                    evaluation_mode="verify",
                ),
                max_samples=1,
                max_generations=1,
                pop_size=1,
            )
            self.assertGreaterEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected_or_failed"], 0)


if __name__ == "__main__":
    unittest.main()
