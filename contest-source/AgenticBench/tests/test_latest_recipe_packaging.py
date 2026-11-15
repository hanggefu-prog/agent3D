"""The installed local adapter must retain its trusted latest-flow recipes."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("setuptools") and importlib.util.find_spec("wheel"),
                     "Wheel build requires the optional packaging test environment")
class LatestRecipePackagingTest(unittest.TestCase):
    def test_wheel_binds_all_four_profiles_without_source_checkout(self):
        with tempfile.TemporaryDirectory(prefix="agenticbench-package-") as temp:
            base = Path(temp)
            source = base / "source"
            shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", ".pytest_cache", "*.egg-info",
                "build", "dist", "outputs", "evolution_runs", "tests"))
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            env.pop("PYTHONPATH", None)
            built = subprocess.run([sys.executable, "-I", "-c",
                "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
                cwd=source, env=env, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stdout[-4000:] + built.stderr[-4000:])
            wheels = list((source / "dist").glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            installed = base / "installed"
            with zipfile.ZipFile(wheels[0]) as wheel:
                from local_bench.tp_gnn_openevolve import TEMPLATES
                for name in TEMPLATES:
                    relative = "local_bench/templates/optimization/" + name
                    self.assertIn(relative, wheel.namelist())
                    self.assertEqual(wheel.read(relative), (ROOT / relative).read_bytes())
                for stage in ("partition", "placement"):
                    relative = "local_bench/reference_configs/" + stage + ".json"
                    self.assertTrue(relative in wheel.namelist(), "Missing wheel resource: " + relative)
                    self.assertEqual(wheel.read(relative), (ROOT / relative).read_bytes())
                wheel.extractall(installed)
            # Remove the build input before exercising the installed adapter.
            shutil.rmtree(source)
            code = """
import json, pathlib, sys, yaml
sys.path.insert(0, sys.argv[1])
from local_bench import reference_latest as latest
from local_bench import OpenEvolveAskTell, run_task_search
root = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(latest.__file__).resolve().is_relative_to(root)
for case in latest.CASES:
    config = yaml.safe_load((latest.profile_path(case) / 'config.yaml').read_text())
    resolved = latest.bind_recipe_paths(config, root / 'not-a-physical-checkout')
    for stage in ('partition', 'placement'):
        path = pathlib.Path(resolved[stage]['config']).resolve()
        assert path.is_relative_to(root)
        assert isinstance(json.loads(path.read_text()), dict)
print('Four installed profiles bind both trusted recipes; no EDA executed.')
"""
            runtime_python = os.environ.get("AGENTICBENCH_INSTALLED_TEST_PYTHON", sys.executable)
            checked = subprocess.run([runtime_python, "-I", "-c", code, str(installed)],
                cwd=base, env=env, text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__ == "__main__":
    unittest.main()
