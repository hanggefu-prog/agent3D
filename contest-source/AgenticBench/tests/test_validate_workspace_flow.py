import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import validate_workspace_flow as probe


@pytest.mark.parametrize("success", [True, False])
def test_reference_only_probe_calls_real_interface_once(tmp_path, monkeypatch, success):
    calls = []
    class Evaluator:
        def __init__(self, runtime, case, objective, bundle, task):
            assert case == "bp_fe" and objective == "thermal"
        def evaluate(self, subject, output, reference):
            calls.append((subject, output, reference))
            return {"success": success, "metrics": {"combined_score": 1.0} if success else {}}
    monkeypatch.setattr(probe, "WorkspaceEvaluator", Evaluator)
    runtime = SimpleNamespace(contained=lambda p: Path(p), source=tmp_path / "source",
        framework=probe.ROOT, source_commit="frozen", image="sha256:fixed")
    output = tmp_path / "probe"
    assert probe.qualify(runtime, "bp_fe", "thermal", tmp_path / "input", output) == (0 if success else 1)
    assert len(calls) == 1 and calls[0][2] is None
    result = json.loads((output / "summary.json").read_text())
    assert result["model_calls"] == result["candidate_finals"] == 0
    with pytest.raises(FileExistsError):
        probe.qualify(runtime, "bp_fe", "thermal", tmp_path / "input", output)
    assert len(calls) == 1
