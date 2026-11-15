from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import yaml

from agentic3d.contracts import LOCAL_TASK_TEMPLATES
from local_bench.model import load_task


ROOT = Path(__file__).resolve().parents[1]


class RepositoryBoundaryTest(unittest.TestCase):
    def test_runtime_harness_parses_with_python_38_grammar(self) -> None:
        for path in sorted((ROOT / "local_bench").glob("*.py")):
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 8),
            )

    def test_open3dbench_is_external(self) -> None:
        forbidden = ["Place-3D", "OpenROAD-3D", "platforms", "benchmarks", "runs"]
        present = [name for name in forbidden if (ROOT / name).exists()]
        self.assertEqual(present, [], "vendored Open3DBench content: {}".format(present))
        lock = json.loads((ROOT / "open3dbench.lock.json").read_text(encoding="utf-8"))
        self.assertFalse(lock["vendored"])
        self.assertFalse(lock["submodule"])
        self.assertEqual(lock["integration"], "external-path-adapter")

    def test_initial_tasks_use_ng45_ten_layer_profile(self) -> None:
        for task in sorted((ROOT / "local_tasks").glob("*/task.yaml")):
            spec = load_task(task.parent)
            config = yaml.safe_load(spec.base_config.read_text(encoding="utf-8"))
            self.assertEqual(config["pdk"], "ng45", str(spec.base_config))
            self.assertNotEqual(config["pdk"], "ng45_6l", str(spec.base_config))

    def test_task_contracts_have_openevolve_files(self) -> None:
        required = {
            "task.yaml",
            "prompt.md",
            "config.yaml",
            "initial_program.py",
        }
        tasks = [ROOT / "local_tasks" / name for name in LOCAL_TASK_TEMPLATES]
        self.assertEqual(len(tasks), 10)
        self.assertNotIn("global_routing_net_ordering", LOCAL_TASK_TEMPLATES)
        for task in tasks:
            self.assertTrue(required.issubset({path.name for path in task.iterdir()}), task.name)
            self.assertTrue((task / "initial_candidate" / "candidate.json").is_file())
            self.assertTrue(load_task(task).base_config.is_file())
            # The exported bundle supplies the common evaluator; no per-task
            # wrapper is required for ask/tell tasks.
            self.assertTrue((ROOT / "local_bench/openevolve_adapter.py").is_file())

    def test_qwen_config_never_contains_a_key_value(self) -> None:
        config = yaml.safe_load((ROOT / "local_bench" / "configs" / "qwen.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["api_key_env"], "DASHSCOPE_API_KEY")
        self.assertNotIn("api_key", config)

    def test_openai_config_never_contains_a_key_value(self) -> None:
        config = yaml.safe_load((ROOT / "local_bench" / "configs" / "openai.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(config["api_mode"], "responses")
        self.assertNotIn("api_key", config)

    def test_openai_compatible_template_is_chat_only_and_secret_free(self) -> None:
        config = yaml.safe_load(
            (
                ROOT
                / "local_bench"
                / "configs"
                / "openai_compatible.example.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["provider"], "openai")
        self.assertEqual(config["api_mode"], "chat_completions")
        self.assertEqual(config["api_key_env"], "OPENAI_COMPAT_API_KEY")
        self.assertTrue(config["base_url"].startswith("https://"))
        self.assertNotIn("api_key", config)

    def test_deepseek_config_never_contains_a_key_value(self) -> None:
        config = yaml.safe_load(
            (ROOT / "local_bench" / "configs" / "deepseek.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["provider"], "deepseek")
        self.assertEqual(config["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(config["api_mode"], "chat_completions")
        self.assertEqual(config["model"], "deepseek-v4-flash")
        self.assertEqual(config["reasoning_effort"], "high")
        self.assertIsNone(config["temperature"])
        self.assertIsNone(config["top_p"])
        self.assertTrue(config["stream"])
        self.assertNotIn("api_key", config)


if __name__ == "__main__":
    unittest.main()
