import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic3d.knowledge import digest, write_json
from agentic3d.sta_protocol import COMMON, METRICS, TASK_SCHEMA, validate_task
from agentic3d.sta_knowledge_runner import cached_call, validate_evidence_citation
from local_bench import load_task


def task():
    return dict(id="timing-closure", description="Reusable closure", stage="partition",
                source_scope=["Place-3D/src/partition/lol/methods/tp_gnn.py"],
                evaluation="partition", objective=[dict(metric="timing_weighted_cut", weight=1, scale_floor=1)],
                procedure="Inspect and test", validation="Trusted metrics")


def test_schema_prevents_observed_id_and_metric_failures():
    pattern = TASK_SCHEMA["properties"]["id"]["pattern"]
    assert re.fullmatch(pattern, task()["id"])
    assert not re.fullmatch(pattern, "tp_gnn_closure")
    choices = TASK_SCHEMA["properties"]["objective"]["items"]["properties"]["metric"]["enum"]
    assert "cut_pin_count" not in choices
    assert set(choices) == set().union(*METRICS.values())
    assert all(metric in COMMON for metrics in METRICS.values() for metric in metrics)
    assert "objective=[]" in COMMON and "max_inner_attempts=1" in COMMON


def test_repair_preserves_response_and_cumulative_errors(tmp_path):
    responses = [dict(hypothesis="preserve-me", value=i) for i in range(3)]
    prompts = []
    def ask(prompt):
        prompts.append(prompt)
        return responses[len(prompts)-1]
    def validate(response):
        if response["value"] < 2:
            raise ValueError(["bad id", "bad metric"][response["value"]])
    path = tmp_path / "plan.json"
    result = cached_call(SimpleNamespace(ask=ask), path, COMMON, validate)
    assert result["value"] == 2
    assert all(x in prompts[2] for x in ("bad id", "bad metric", "preserve-me", '"value": 1'))
    assert json.loads(path.with_suffix('.validation.json').read_text())["errors"] == ["bad id", "bad metric"]
    cached_call(SimpleNamespace(ask=lambda _: pytest.fail("cache should not call model")), path, COMMON, validate)


def test_evidence_original_path_resolves_only_unique_checked_local_file(tmp_path):
    original = "20_partition/tp_gnn_report.json"
    local = tmp_path / "evidence/reference/00.txt"
    local.parent.mkdir(parents=True); local.write_text("verified tail")
    write_json(local.parent / "index.json", {"files": [dict(path=original, local_file=str(local), sha256=digest("verified tail"))]})
    assert validate_evidence_citation(original, tmp_path) == local.resolve()
    assert validate_evidence_citation("evidence/reference/00.txt", tmp_path) == local.resolve()
    with pytest.raises(ValueError): validate_evidence_citation("../other-run/log", tmp_path)
    local.write_text("tampered")
    with pytest.raises(ValueError): validate_evidence_citation(original, tmp_path)
    local.write_text("verified tail")
    other = tmp_path / "evidence/round-1/00.txt"
    other.parent.mkdir(); other.write_text("verified tail")
    write_json(other.parent / "index.json", {"files": [dict(path=original, local_file=str(other), sha256=digest("verified tail"))]})
    with pytest.raises(ValueError): validate_evidence_citation(original, tmp_path)


def test_final_and_cross_primitive_guards_unchanged():
    spec = load_task(
        Path(__file__).resolve().parents[1]
        / "evaluation_profiles/tp_gnn_placement_two_pass"
    )
    candidate = task(); candidate["evaluation"] = "final"
    with pytest.raises(ValueError, match="immutable"): validate_task(candidate, spec)
    candidate["objective"] = []
    validate_task(candidate, spec)
    candidate = task(); candidate["objective"][0]["metric"] = "placement_native_hpwl"
    with pytest.raises(ValueError, match="allowed:"): validate_task(candidate, spec)
