"""Catalog CLI works outside the checkout and never dispatches an evaluator."""
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def test_cli_binds_trusted_packages_outside_checkout(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/reference_catalog.py"
    producer = tmp_path / "producer.json"
    producer.write_text(json.dumps({"kind": "native-evaluation-v1", "case": "bp_fe", "endpoint": "timing",
        "profile": "tp_gnn_placement_one_pass", "run_root": str(tmp_path / "missing-run")}))
    env = {**os.environ, "PYTHONPATH": str(Path(yaml.__file__).resolve().parent.parent)}
    result = subprocess.run([sys.executable, str(script), "qualify", "--producer", str(producer)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "missing-run" in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["producer.json"]
