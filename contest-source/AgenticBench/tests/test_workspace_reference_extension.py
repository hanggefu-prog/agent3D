"""Immutable suffix preparation with schema-complete fixtures, not EDA."""
import copy
import json
from pathlib import Path

import pytest

from local_bench import reference_catalog as catalog, reference_evaluated as native
from local_bench.reference_store import file_sha
from scripts.workspace_reference_extension import qualify_parent, copy_qualified_prefix
from test_reference_evaluated import make_evaluation


def registered(make_evaluation, tmp_path, endpoint="timing", *, rounds=5):
    from scripts.workspace_task_selection import workspace_task_id
    output, run, _ = make_evaluation(endpoint, rounds=rounds)
    profile, task_id, _, _ = native.ENDPOINTS[endpoint]
    task_id = workspace_task_id(native.ENDPOINTS[endpoint][2], rounds)
    producer = {"kind": "native-evaluation-v1", "case": "bp_fe", "endpoint": endpoint,
                "profile": profile, "evaluation_root": str(output), "run_root": str(run)}
    payload = catalog.qualify_producer(producer)
    contract = catalog.make_contract(payload,
        task_digest=catalog.task_digest(native.FRAMEWORK / "global_tasks" / task_id),
        evaluator_digest="e" * 64)
    record = catalog.register_producer(tmp_path / "store", producer, origin="FRESH", contract=contract)
    return record, contract, run


def target(parent, endpoint):
    profile, task_id, _, _ = native.ENDPOINTS[endpoint]
    physical = parent["physical"]
    return {**parent, "endpoint": endpoint,
        "task_digest": catalog.task_digest(native.FRAMEWORK / "global_tasks" / task_id),
        "physical": native.expected_source_contract(parent["case"], endpoint, profile=profile,
            source_revision=physical["source_revision"], image=physical["image"])}


@pytest.mark.parametrize("parent_endpoint,endpoint,start", [
    ("timing", "cts_thermal", "cts"), ("timing", "routed_thermal", "cts"),
    ("cts_thermal", "routed_thermal", "global_routing")])
def test_copy_complete_prefix_without_executing_or_relabeling(make_evaluation, tmp_path, parent_endpoint, endpoint, start):
    record, contract, run = registered(make_evaluation, tmp_path, parent_endpoint)
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    parent = qualify_parent(record, target(contract, endpoint), record_sha256=file_sha(record))
    assert parent["start_stage"] == start
    destination = tmp_path / "extension" / "run"
    inventory = copy_qualified_prefix(parent, destination)
    assert [row["stage"] for row in inventory] == parent["stages"]
    for row in inventory:
        relative = Path(native.verifier.STAGE_DIRS[row["stage"]]) / "stage_manifest.json"
        assert (destination / relative).read_bytes() == (run / relative).read_bytes()
    assert not (destination / "95_thermal_snapshot").exists()
    assert not (destination / "75_global_route").exists()
    assert set(json.loads((destination / "run.json").read_text())["stages"]) == set(parent["stages"])
    assert before == {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="cannot be retried"):
        copy_qualified_prefix(parent, destination)


def test_ten_round_parent_keeps_its_own_authority_when_extending_physical_suffix(make_evaluation, tmp_path):
    record, contract, run = registered(make_evaluation, tmp_path, rounds=10)
    parent = qualify_parent(record, target(contract, "cts_thermal"), record_sha256=file_sha(record))
    assert parent["record_value"]["contract"]["task_digest"] == contract["task_digest"]
    assert parent["start_stage"] == "cts"
    before = record.read_bytes()
    copy_qualified_prefix(parent, tmp_path / "child/run")
    assert record.read_bytes() == before


@pytest.mark.parametrize("mutation", ["source", "image", "input", "evaluator"])
def test_parent_contract_mismatch_cannot_extend(make_evaluation, tmp_path, mutation):
    record, contract, _ = registered(make_evaluation, tmp_path)
    desired = copy.deepcopy(target(contract, "cts_thermal"))
    if mutation == "source":
        desired["physical"]["source_revision"] = "2" * 40
    elif mutation == "image":
        desired["physical"]["image"] = next(i for i in native.baseline.IMAGE_IDS if i != desired["physical"]["image"])
    elif mutation == "input":
        desired["frozen_inputs"]["benchmark_input_manifest_sha256"] = "b" * 64
    else:
        desired["evaluator_digest"] = "b" * 64
    with pytest.raises(ValueError, match="contract mismatch"):
        qualify_parent(record, desired)


def test_record_or_prefix_change_rejected_before_copy(make_evaluation, tmp_path):
    record, contract, run = registered(make_evaluation, tmp_path)
    parent = qualify_parent(record, target(contract, "cts_thermal"))
    manifest = run / "60_opt/stage_manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    destination = tmp_path / "extension"
    with pytest.raises(ValueError, match="authenticated prefix manifest"):
        copy_qualified_prefix(parent, destination)
    assert not destination.exists()
    record.chmod(0o644)
    record.write_text(record.read_text() + " ")
    with pytest.raises(ValueError, match="changed after selection"):
        qualify_parent(record, target(contract, "cts_thermal"), record_sha256=parent["record_sha256"])


def test_no_same_endpoint_or_historical_bridge(make_evaluation, tmp_path):
    record, contract, _ = registered(make_evaluation, tmp_path)
    with pytest.raises(ValueError, match="missing suffix"):
        qualify_parent(record, contract)
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"producer": {"kind": "frozen-20260923-v1"}}))
    with pytest.raises(ValueError, match="historical source composition"):
        qualify_parent(historical, target(contract, "cts_thermal"))
