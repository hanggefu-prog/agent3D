"""Reference storage tests use synthetic evidence, never native qualification."""
import copy
import json
from pathlib import Path

import pytest

from local_bench import reference_store as store


@pytest.fixture
def reference(tmp_path):
    run = tmp_path / "producer" / "run"
    run.mkdir(parents=True)
    artifact = run / "native.log"
    artifact.write_text("fixture measurement 4.5\n")
    physical = {"opt_passes": 1, "threads": 4, "image": "fixture"}
    contract = {"case": "bp_fe", "endpoint": "timing", "task_digest": "a" * 64,
                "evaluator_digest": "b" * 64, "frozen_inputs": {"floorplan": "c" * 64},
                "physical": physical}
    payload = {"qualified": True, "case": "bp_fe", "endpoint": "timing",
               "frozen_inputs": contract["frozen_inputs"],
               "source_contract": physical, "source_identity": {"native_source": "fixture"},
               "baseline": {"raw_metrics": {"wns_setup_ns": -4.5}, "source_run": str(run),
                            "native_runtime_historical": {"opt": 25.0}},
               "evidence": [{"path": str(artifact), "bytes": artifact.stat().st_size,
                             "sha256": store.file_sha(artifact), "role": "native-log"}]}
    calls = []
    def validate(producer):
        calls.append(producer)
        if artifact.read_text() != "fixture measurement 4.5\n":
            raise ValueError("Independent fixture validator rejected changed measurement")
        return copy.deepcopy(payload)
    return dict(contract=contract, payload=payload, validate=validate, calls=calls,
                producer={"kind": "fixture"}, root=tmp_path / "registry", artifact=artifact)


def publish(ref, origin="ARCHIVE"):
    producer = dict(ref["producer"])
    if origin == "EXTEND":
        producer["parent_reference_id"] = "d" * 64
    return store.publish_reference(ref["root"], contract=ref["contract"], origin=origin,
                                   producer=producer, validate=ref["validate"])


@pytest.mark.parametrize("origin", ["ARCHIVE", "FRESH", "EXTEND"])
def test_all_qualified_origins_can_be_adopted(reference, tmp_path, origin):
    path = publish(reference, origin)
    assert path.stat().st_mode & 0o222 == 0
    receipt = store.adopt_reference(path, contract=reference["contract"],
                                   validate=reference["validate"], output=tmp_path / "adoption")
    assert receipt["original_origin"] == origin
    assert receipt["new_reference_native_attempts"] == 0
    assert receipt["model_calls"] == receipt["candidate_finals"] == 0
    assert receipt["historical_native_runtime"] == {"opt": 25.0}
    assert len(reference["calls"]) == 2


def test_publication_is_idempotent_without_rewriting_record(reference):
    first = publish(reference)
    content, timestamp = first.read_bytes(), first.stat().st_mtime_ns
    assert publish(reference) == first
    assert first.read_bytes() == content and first.stat().st_mtime_ns == timestamp


@pytest.mark.parametrize("field,value", [("case", "bp"), ("endpoint", "cts_thermal"),
    ("task_digest", "d" * 64), ("evaluator_digest", "e" * 64),
    ("frozen_inputs", {"floorplan": "f" * 64}), ("physical", {"opt_passes": 2})])
def test_contract_mismatch_never_adopts(reference, field, value):
    path = publish(reference)
    contract = {**reference["contract"], field: value}
    with pytest.raises(ValueError, match="contract mismatch"):
        store.load_reference(path, contract=contract, validate=reference["validate"])


def test_failed_adoption_is_preserved_and_cannot_retry_in_place(reference, tmp_path):
    path = publish(reference)
    reference["artifact"].write_text("tampered measurement\n")
    output = tmp_path / "adoption"
    with pytest.raises(ValueError, match="artifact changed"):
        store.adopt_reference(path, contract=reference["contract"], validate=reference["validate"], output=output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["new_reference_native_attempts"] == 0
    with pytest.raises(FileExistsError):
        store.adopt_reference(path, contract=reference["contract"], validate=reference["validate"], output=output)


def test_requalification_not_only_a_passed_flag(reference):
    path = publish(reference)
    def reject(_):
        raise ValueError("native provenance rejected")
    with pytest.raises(ValueError, match="native provenance"):
        store.load_reference(path, contract=reference["contract"], validate=reject)


def test_payload_change_rejected_even_with_unchanged_artifact(reference):
    path = publish(reference)
    reference["payload"]["baseline"]["raw_metrics"]["wns_setup_ns"] = 0
    with pytest.raises(ValueError, match="requalification differs"):
        store.load_reference(path, contract=reference["contract"], validate=reference["validate"])


def test_no_reference_is_not_an_instruction_to_run_fresh(reference):
    with pytest.raises(ValueError, match="no automatic"):
        store.find_reference(reference["root"], contract=reference["contract"])
    assert reference["calls"] == []


def test_multiple_references_require_explicit_selection(reference):
    path = publish(reference)
    publish(reference, "FRESH")
    with pytest.raises(ValueError, match="found 2"):
        store.find_reference(reference["root"], contract=reference["contract"])
    assert store.find_reference(reference["root"], contract=reference["contract"], reference_id=path.stem) == path


@pytest.mark.parametrize("invalid", [False, None, 0])
def test_failed_or_uncertain_result_is_not_publishable(reference, invalid):
    reference["payload"]["qualified"] = invalid
    with pytest.raises(ValueError, match="independently qualify"):
        publish(reference, "FRESH")
    assert not reference["root"].exists()


def test_extend_requires_parent(reference):
    with pytest.raises(ValueError, match="parent"):
        store.publish_reference(reference["root"], contract=reference["contract"], origin="EXTEND",
                                producer=reference["producer"], validate=reference["validate"])


def test_nan_is_not_publishable(reference):
    reference["payload"]["baseline"]["raw_metrics"]["wns_setup_ns"] = float("nan")
    with pytest.raises(ValueError):
        publish(reference)


def test_symlink_evidence_is_not_publishable(reference):
    link = reference["artifact"].parent / "link"
    link.symlink_to(reference["artifact"])
    reference["payload"]["evidence"][0]["path"] = str(link)
    with pytest.raises(ValueError, match="non-symlink"):
        publish(reference)


def test_store_cannot_be_inside_physical_run(reference):
    reference["root"] = reference["artifact"].parent / "registry"
    with pytest.raises(ValueError, match="disjoint"):
        publish(reference)


@pytest.mark.parametrize("field", ["reference_root", "thermal_root", "evaluation_root", "run_root"])
def test_store_cannot_pollute_archive_root_before_qualification(reference, tmp_path, field):
    protected = tmp_path / "sealed"
    protected.mkdir()
    reference["producer"][field] = str(protected)
    reference["root"] = protected / "registry"
    with pytest.raises(ValueError, match="disjoint"):
        publish(reference)
    assert not reference["root"].exists()
    assert reference["calls"] == []


@pytest.mark.parametrize("location", ["archive", "run", "store", "store-sibling"])
def test_adoption_cannot_write_attempt_inside_authority(reference, tmp_path, location):
    archive = tmp_path / "archive"
    archive.mkdir()
    reference["producer"]["reference_root"] = str(archive)
    path = publish(reference)
    roots = {"archive": archive, "run": reference["artifact"].parent, "store": path.parent,
             "store-sibling": path.parent.parent}
    output = roots[location] / "new-attempt"
    with pytest.raises(ValueError, match="disjoint"):
        store.adopt_reference(path, contract=reference["contract"], validate=reference["validate"], output=output)
    assert not output.exists()


def test_origin_cannot_relabel_known_native_producer(reference):
    reference["producer"] = {"kind": "native-evaluation-v1"}
    path = publish(reference, "FRESH")
    raw = json.loads(path.read_text())
    raw["origin"] = "ARCHIVE"
    body = {key: raw[key] for key in ("schema", "contract", "origin", "producer", "qualification")}
    raw["reference_id"] = store.digest(body)
    forged = path.parent / (raw["reference_id"] + ".json")
    forged.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="origin differs"):
        store.load_reference(forged, contract=reference["contract"], validate=reference["validate"])


def test_store_rejects_symlink_destination(reference, tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    link = tmp_path / "link"
    link.symlink_to(protected, target_is_directory=True)
    reference["root"] = link / "registry"
    with pytest.raises(ValueError, match="symlinks"):
        publish(reference)
    assert not (protected / "registry").exists()


@pytest.mark.parametrize("target_kind", ["archive", "physical-run"])
def test_store_rejects_symlink_records_directory(reference, tmp_path, target_kind):
    archive = tmp_path / "archive"
    archive.mkdir()
    reference["producer"]["reference_root"] = str(archive)
    target = archive if target_kind == "archive" else reference["artifact"].parent
    before = sorted(p.name for p in target.iterdir())
    reference["root"].mkdir()
    (reference["root"] / "records").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        publish(reference)
    assert sorted(p.name for p in target.iterdir()) == before


def test_store_rejects_dangling_record_symlink(reference, tmp_path):
    body = {"schema": store.SCHEMA, "contract": reference["contract"], "origin": "ARCHIVE",
            "producer": reference["producer"], "qualification": reference["payload"]}
    records = reference["root"] / "records"
    records.mkdir(parents=True)
    target = tmp_path / "must-not-be-created.json"
    (records / (store.digest(body) + ".json")).symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        publish(reference)
    assert not target.exists()
