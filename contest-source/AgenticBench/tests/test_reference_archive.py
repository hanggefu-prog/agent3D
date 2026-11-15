"""Producer adoption tests: real file/hash checks, no native or model calls."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from local_bench import reference_archive as subject
from local_bench import reference_baseline as baseline


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def file_record(file, **extra):
    return {"bytes": file.stat().st_size, "sha256": baseline.sha256(file), **extra}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    run = root / "user_runs/original/bp_fe/run"
    records = {}
    for stage, directory in subject.STAGES.items():
        stage_root = run / directory
        stage_root.mkdir(parents=True)
        write(stage_root / "metrics.json", {"native": stage})
        records[stage] = {"status": "success", "directory": directory, "runtime_seconds": 1.5}
    (run / "config.yaml").write_text("design: bp_fe\ntype: lol\npdk: ng45\nthreads: 4\n")
    (run.parent / "config.yaml").write_text("design: bp_fe\ntype: lol\npdk: ng45\n")
    (run / "10_floorplan/design.def").write_text("frozen DEF\n")
    write(run / "10_floorplan/io_metrics.json", {"seed": 1})
    part = run / "20_partition"
    (part / "partition_assignment.csv").write_text("inst_name,die\na,bottom\nb,top\n")
    native = {"method": "tp-gnn-flat", "config": {"decoder": "global-2way", "seed": 1001},
              "hierarchy": {"mode": "flat", "consumed_as_feature": False}}
    write(part / "tp_gnn_report.json", native)
    write(part / "partition_manifest.json", {"method": "tp-gnn-flat", "seed": 1001, "method_report": native})
    write(part / "partition_metrics.json", {"instance_count": 2, "bottom_count": 1,
          "top_count": 1, "area_balance_error": 0, "cut_net_count": 1})
    write(part / "timing_input/timing_manifest.json", {"producer": "legacy-stage-20"})
    place = run / "30_placement"
    write(place / "metrics.json", {"dies": {die: {"missing_connection_object_count": 0}
          for die in ("bottom", "top")}, "total_object_origin_hpwl_um": 45.0})
    gp = place / "stages/02_global_place"
    write(gp / "stage_manifest.json", {"returncode": 0, "initialization_seed": 1000})
    write(gp / "params.json", {"global_place_stages": [{"optimizer": "nesterov", "iteration": 4000}]})
    (gp / "placement.gp.def").write_text("native global DEF\n")
    (gp / "run.log").write_text("wHPWL 200 Overflow [0.2 0.3] MaxDensity [0.9 1.0]\n"
                                "wHPWL 100 Overflow [0.1 0.2] MaxDensity [0.7 0.8]\n"
                                "legalization HPWL 123\n")
    previous = None
    for stage, directory in list(subject.STAGES.items())[3:]:
        current = run / directory
        cp = {"schema_version": 1, "pdk": "ng45", "stage": stage, "dies": {}}
        for die in ("bottom", "top"):
            cp["dies"][die] = {}
            for kind, ext in (("odb", "odb"), ("def", "def"), ("verilog", "v"), ("sdc", "sdc")):
                path = current / die / ("design." + ext)
                path.parent.mkdir(exist_ok=True)
                path.write_text("native " + directory + " " + die + " " + ext)
                cp["dies"][die][kind] = path.relative_to(current).as_posix()
        if previous:
            cp["source_checkpoint"] = "../" + previous + "/checkpoint.json"
        if stage == "cts":
            cp["ready_for_route"] = True
        write(current / "checkpoint.json", cp)
        previous = directory
    final = {"wns_setup": -0.6, "tns_setup": -12.0, "sta_completeness_checks_passed": 1.0, "passive_hbt_rc": 1.0}
    write(run / "60_opt/metrics.json", {
        "status": "success", "algorithm": "openroad-coordinated-passive-cutpoint",
        "passes_run": 1, "iterations_run": 1, "die_passes": {"bottom": 1, "top": 1},
        "pass_record": {"pass": 1, "die_order": ["bottom", "top"],
                        "die_metrics": {"bottom": {"repair": 1}, "top": {"repair": 1}},
                        "constraints": {"bottom": {}, "top": {}}, "global_after": final},
        "final_global": final, "final_boundary_coverage": {"sta_unconstrained_endpoints": 0},
        "timing_validity": {"authoritative_timing_view": "continuous-passive-hbt", "graph_completeness": "passed",
            "die_internal_parasitics_in_global_sta": True, "hbt_is_passive_rc_in_authoritative_global_sta": True,
            "remote_die_load_visible_to_source_driver": True, "unclocked_or_unconstrained_endpoints": 0,
            "missing_liberty_cells": 0, "combinational_loops": 0}})
    write(run / "70_cts/metrics.json", {"status": "success", "algorithm": "openroad-balanced"})
    write(run / "run.json", {"design": "bp_fe", "type": "lol", "pdk": "ng45", "status": "failed",
          "stages": {**records, "routing": {"status": "failed", "directory": "80_route"}}})
    write(root / "metadata/source_provenance.json", {"producer": "historical"})
    write(root / "metadata/image_identity.json", {"native_tools": "historical-fixture"})
    case_record = {"config_sha256": baseline.sha256(run.parent / "config.yaml"),
                   "work": run.parent.relative_to(root).as_posix()}
    index = {"reference_id": "fixture-reference", "original_root": "/original",
             "source_provenance": "metadata/source_provenance.json", "cases": {"bp_fe": case_record},
             "image": {"full_layer_and_tool_evidence": "metadata/image_identity.json"}}
    identity = {"reference_id": index["reference_id"], "cases": {"bp_fe": {
        "run": run.relative_to(root).as_posix(),
        "physical_prefix": {"10_floorplan/design.def": file_record(run / "10_floorplan/design.def")},
        "partition_assignment_sha256": baseline.sha256(part / "partition_assignment.csv")}}}

    def seal():
        for stage, directory in subject.STAGES.items():
            current = run / directory
            artifacts = {path.relative_to(current).as_posix(): file_record(path, path=path.relative_to(current).as_posix())
                         for path in current.rglob("*") if path.is_file() and path != current / "stage_manifest.json"}
            artifacts["method_manifest"] = artifacts.pop("partition_manifest.json") if stage == "partition" else None
            if stage != "partition":
                artifacts.pop("method_manifest")
            write(current / "stage_manifest.json", {"schema_version": 1, "status": "success", "stage": stage,
                  "method": "tp-gnn-flat" if stage == "partition" else "native", "signature": {"producer": "frozen"},
                  "runtime_seconds": 2.5, "artifacts": artifacts})
        write(root / "reference.json", index)
        identity["reference_json_sha256"] = baseline.sha256(root / "reference.json")
        inventory = {path.relative_to(root).as_posix(): file_record(path, kind="file")
                     for path in root.rglob("*") if path.is_file() and path.name != "FILES.json"}
        write(root / "FILES.json", inventory)
        monkeypatch.setattr(subject, "FILES_SHA256", baseline.sha256(root / "FILES.json"))
        monkeypatch.setattr(baseline, "identity", lambda: copy.deepcopy(identity))
    seal()
    return {"root": root, "run": run, "index": index, "seal": seal, "tmp": tmp_path}


def prepare(frozen, endpoint="placement", **kwargs):
    return subject.prepare_archive_reference(frozen["root"], "bp_fe", endpoint, **kwargs)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_readonly_native_placement_not_geometric_proxy(frozen, monkeypatch):
    before = snapshot(frozen["root"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No subprocess is allowed"))
    row = prepare(frozen)
    assert row["baseline"]["raw_metrics"]["placement_native_hpwl"] == 100
    assert row["baseline"]["raw_metrics"]["placement_overflow_max"] == .2
    assert row["baseline"]["source_run_status"] == "failed"  # later DRT remains failed
    assert row["qualified"] and not row["new_hook_equivalence_qualified"]
    assert not row["current_runtime_equivalence_qualified"]
    assert row["executed_native_stages"] == []
    assert row["candidate_finals"] == row["new_model_calls"] == 0
    assert all(Path(item["path"]).is_absolute() for item in row["evidence"])
    assert row == prepare(frozen)  # deterministic, no validation wall time/timestamp
    assert before == snapshot(frozen["root"])


def test_timing_reads_one_pass_native_evidence_without_fake_stage15(frozen):
    row = prepare(frozen, "timing")
    assert row["baseline"]["raw_metrics"]["wns_setup_ns"] == -.6
    assert row["baseline"]["raw_metrics"]["tns_deficit"] == 12
    assert not (frozen["run"] / "15_timing_extract").exists()


def test_requested_and_normalized_config_have_distinct_authenticated_identities(frozen):
    row = prepare(frozen)
    identity = row["source_identity"]
    assert identity["requested_config_sha256"] == baseline.sha256(frozen["run"].parent / "config.yaml")
    assert identity["normalized_executed_config_sha256"] == baseline.sha256(frozen["run"] / "config.yaml")
    assert identity["requested_config_sha256"] != identity["normalized_executed_config_sha256"]
    frozen["index"]["cases"]["bp_fe"]["config_sha256"] = identity["normalized_executed_config_sha256"]
    frozen["seal"]()
    with pytest.raises(ValueError, match="requested configuration"):
        prepare(frozen)


def test_historical_runtime_uses_native_manifests_not_ledger(frozen):
    timing = prepare(frozen, "timing")["baseline"]["native_runtime_historical"]
    assert timing["total_component_seconds"] == 7 * 2.5
    assert timing["components"]["60_opt"] == 2.5  # ledger says 1.5
    assert "80_route" not in timing["components"]
    assert timing["not_contiguous_elapsed_time"] is True


def test_missing_historical_runtime_is_unknown_not_zero():
    value = subject._historical_runtime({"placement": {"directory": "30_placement", "runtime_seconds": None}}, "placement")
    assert value["components"]["30_placement"] is None
    assert value["unknown_components"] == ["30_placement"]
    assert value["total_component_seconds"] is None
    assert value["sum_recorded_component_seconds"] is None


@pytest.mark.parametrize("value", [-1, float("nan"), "2", True])
def test_malformed_historical_runtime_is_rejected(value):
    with pytest.raises(ValueError):
        subject._historical_runtime({"placement": {"directory": "30_placement", "runtime_seconds": value}}, "placement")


@pytest.mark.parametrize("relative", ["reference.json", "FILES.json", "metadata/source_provenance.json",
    "user_runs/original/bp_fe/run/30_placement/stages/02_global_place/run.log",
    "user_runs/original/bp_fe/run/30_placement/stage_manifest.json"])
def test_tampered_archive_is_rejected(frozen, relative):
    path = frozen["root"] / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        prepare(frozen)


def test_finite_native_metrics_are_required(frozen):
    path = frozen["run"] / "30_placement/stages/02_global_place/run.log"
    path.write_text("wHPWL 100 Overflow [NaN 0.2] MaxDensity [0.7 0.8]\n")
    frozen["seal"]()
    with pytest.raises(ValueError, match="nonfinite"):
        prepare(frozen)


def test_wrong_checkpoint_chain_is_rejected_even_with_updated_fixture_anchor(frozen):
    path = frozen["run"] / "60_opt/checkpoint.json"
    row = json.loads(path.read_text())
    row["source_checkpoint"] = "../30_placement/checkpoint.json"
    write(path, row)
    frozen["seal"]()
    with pytest.raises(ValueError, match="predecessor"):
        prepare(frozen, "timing")


def test_pure_summary_cannot_replace_full_native_proofs(frozen):
    (frozen["run"] / "60_opt/bottom/design.odb").unlink()
    with pytest.raises(ValueError):
        prepare(frozen, "timing")


@pytest.mark.parametrize("endpoint", ["routed_thermal", "report", "signoff", "75_global_route"])
def test_unsupported_endpoint_fails_closed(frozen, endpoint):
    with pytest.raises(ValueError, match="Unsupported"):
        prepare(frozen, endpoint)


def test_missing_thermal_archive_is_not_zero_temperature(frozen):
    with pytest.raises(ValueError, match="full separately retained"):
        prepare(frozen, "cts_thermal")


def test_symlinked_metadata_cannot_escape_archive(frozen):
    path = frozen["root"] / "metadata/source_provenance.json"
    outside = frozen["tmp"] / "outside.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        prepare(frozen)


def test_grt_is_its_native_endpoint_not_drt_or_stage75(frozen):
    root, run = frozen["root"], frozen["run"]
    grt = root / "maintenance/projected/grt/bp_fe"
    grt.mkdir(parents=True)
    for die in ("bottom", "top"):
        path = grt / die / "global_route.odb"
        path.parent.mkdir()
        path.write_text("routed " + die)
        write(grt / die / "grt_manifest.json", {"artifacts": {"global_route.odb": {
            "size_bytes": path.stat().st_size, "sha256": baseline.sha256(path)}}})
    metrics = {"stage": "global_route", "status": "success", "ready_for_extract": False,
               "grt_total_overflow": 0, "grt_wire_length_um": 20,
               "dies": {die: {"grt_total_overflow": 0, "grt_wire_length_um": 10} for die in ("bottom", "top")},
               "configuration": {"threads": 4, "congestion_iterations": 4, "resource_reduction": 0}}
    write(grt / "grt_metrics.json", metrics)
    cp = {"status": "success", "ready_for_extract": False, "ready_for_detailed_route": True,
          "source_checkpoint": "/original/" + (run / "70_cts/checkpoint.json").relative_to(root).as_posix(),
          "source_checkpoint_sha256": baseline.sha256(run / "70_cts/checkpoint.json"),
          "artifacts": {"grt_metrics.json": {"size_bytes": (grt / "grt_metrics.json").stat().st_size,
                         "sha256": baseline.sha256(grt / "grt_metrics.json")}}}
    write(grt / "global_route_checkpoint.json", cp)
    frozen["index"]["cases"]["bp_fe"].update(
        qualified_grt_checkpoint=(grt / "global_route_checkpoint.json").relative_to(root).as_posix(),
        qualified_grt_checkpoint_sha256=baseline.sha256(grt / "global_route_checkpoint.json"), grt_metrics=metrics)
    frozen["seal"]()
    row = prepare(frozen, "grt")
    assert row["baseline"]["raw_metrics"]["grt_wire_length_um"] == 20
    assert not (run / "75_global_route").exists()


@pytest.fixture
def thermal(frozen, monkeypatch):
    root = frozen["tmp"] / "thermal"
    case_root = root / "bp_fe"
    view = "/historical/thermal/bp_fe/input-view/checkpoint.json"
    write(case_root / "input-view/checkpoint.json", {"stage": "cts"})
    for name in ("power.ptrace", "temperature.steady", "bottom/instance_power.rpt", "top/instance_power.rpt"):
        path = case_root / "thermal" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("native thermal evidence\n")
    metrics = {"stage": "thermal_snapshot", "status": "success", "algorithm": "hotspot-steady-grid",
        "source_stage": "cts", "warning_count": 0, "grid_size": 10, "ambient_k": 318.15,
        "power": {"activity_model": "vectorless-primary-input-0.1-placement-estimated-rc",
                  "provenance": {"checkpoint": view, "model": "die-local-openroad-placement-estimated", "threads": 2},
                  "total_w": .1, "conservation_error_w": 1e-14},
        "power_extraction": {die: {"warning_count": 0, "instance_count": 1} for die in ("bottom", "top")},
        "temperature": {"global_average_k": 322, "bottom": {"tile_count": 100, "average_k": 321},
                        "top": {"tile_count": 100, "average_k": 323}}, "stack": {"kind": "provisional"}}
    write(case_root / "thermal/metrics.json", metrics)
    write(case_root / "thermal/snapshot_manifest.json", {"schema_version": 1, "status": "success", "source_stage": "cts",
          "source_checkpoint": view, "grid_size": 10, "ambient_k": 318.15, "metrics": "metrics.json"})
    receipt = {"schema_version": 1, "status": "success", "role": "diagnostic-cts-vectorless-placement-rc-thermal-snapshot",
               "settings": {"ambient_k": 318.15, "grid_size": 10, "input_activity": .1, "threads": 2},
               "input_view_checkpoint": {"path": view}, "audit": {"metrics": metrics},
               "implementation": {"thermal": "frozen"}, "tools": {"openroad": "frozen"}}
    for key, relative in (("original_checkpoint", "70_cts/checkpoint.json"),
                          ("original_manifest", "70_cts/stage_manifest.json"),
                          ("upstream_opt_checkpoint", "60_opt/checkpoint.json")):
        receipt[key] = file_record(frozen["run"] / relative)
    def seal_thermal():
        receipt["outputs"] = {path.relative_to(case_root).as_posix(): file_record(path)
                              for path in case_root.rglob("*") if path.is_file() and path.name != "receipt.json"}
        write(case_root / "receipt.json", receipt)
        monkeypatch.setitem(subject.THERMAL_RECEIPT_SHA256, "bp_fe", baseline.sha256(case_root / "receipt.json"))
    seal_thermal()
    return {"root": root, "case": case_root, "receipt": receipt, "seal": seal_thermal}


def test_separate_cts_snapshot_keeps_original_provenance_and_thermal_threads(frozen, thermal):
    before = snapshot(thermal["root"])
    row = prepare(frozen, "cts_thermal", thermal_root=thermal["root"])
    assert row["baseline"]["raw_metrics"]["global_average_temperature_k"] == 322
    assert row["baseline"]["raw_metrics"]["cts_algorithm"] == "openroad-balanced"
    assert row["source_contract"]["thermal_settings"]["threads"] == 2
    assert row["source_contract"]["physical_threads"] == 4
    assert before == snapshot(thermal["root"])


def test_small_thermal_evidence_subset_is_not_sufficient(frozen, thermal):
    (thermal["case"] / "thermal/bottom/instance_power.rpt").unlink()
    with pytest.raises(ValueError, match="Missing regular archive file"):
        prepare(frozen, "cts_thermal", thermal_root=thermal["root"])


def test_thermal_requires_matching_original_physical_input(frozen, thermal):
    thermal["receipt"]["original_checkpoint"]["sha256"] = "0" * 64
    thermal["seal"]()
    with pytest.raises(ValueError, match="different physical input"):
        prepare(frozen, "cts_thermal", thermal_root=thermal["root"])


def test_thermal_wrong_measurement_even_in_an_independently_resealed_fixture_is_rejected(frozen, thermal):
    path = thermal["case"] / "thermal/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["temperature"]["global_average_k"] = 0
    thermal["receipt"]["audit"]["metrics"] = metrics
    write(path, metrics)
    thermal["seal"]()
    with pytest.raises(ValueError, match="mean/power"):
        prepare(frozen, "cts_thermal", thermal_root=thermal["root"])


def archive_publication(frozen, *, thermal=None, endpoint="placement"):
    from local_bench import reference_catalog as catalog
    producer = {"kind": subject.PRODUCER_PROFILE, "reference_root": str(frozen["root"]),
                "case": "bp_fe", "endpoint": endpoint,
                "thermal_root": str(thermal["root"]) if thermal else None}
    qualified = catalog.qualify_producer(producer)
    contract = catalog.make_contract(qualified, task_digest="1" * 64, evaluator_digest="2" * 64)
    return catalog, producer, qualified, contract


def test_catalog_archival_adoption_revalidates_without_new_native_attempts(frozen, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No process is allowed"))
    before = snapshot(frozen["root"])
    catalog, producer, qualified, contract = archive_publication(frozen)
    store = frozen["tmp"] / "registry"
    record = catalog.register_producer(store, producer, origin="ARCHIVE", contract=contract)
    assert catalog.register_producer(store, producer, origin="ARCHIVE", contract=contract) == record
    adopted = catalog.adopt(record, contract=contract, output=frozen["tmp"] / "new-attempt")
    assert adopted["original_origin"] == "ARCHIVE"
    assert adopted["reference_origin"] == "adopted"
    assert adopted["new_reference_native_attempts"] == adopted["model_calls"] == adopted["candidate_finals"] == 0
    assert adopted["historical_native_runtime"] == qualified["baseline"]["native_runtime_historical"]
    assert adopted["qualification"] == qualified
    assert snapshot(frozen["root"]) == before


def test_catalog_adoption_records_tampered_archive_failure_without_fallback(frozen):
    catalog, producer, _, contract = archive_publication(frozen)
    record = catalog.register_producer(frozen["tmp"] / "registry", producer, origin="ARCHIVE", contract=contract)
    path = frozen["run"] / "30_placement/stages/02_global_place/placement.gp.def"
    path.write_text("mutated native physical result\n")
    out = frozen["tmp"] / "failed-attempt"
    with pytest.raises(ValueError, match="artifact changed"):
        catalog.adopt(record, contract=contract, output=out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["new_reference_native_attempts"] == 0
    assert failure["model_calls"] == failure["candidate_finals"] == 0
    assert not (out / "reference_adoption.json").exists()
    before = snapshot(out)
    with pytest.raises(FileExistsError):
        catalog.adopt(record, contract=contract, output=out)
    assert snapshot(out) == before


def test_catalog_rejects_frozen_input_mismatch_and_false_fresh_origin(frozen):
    catalog, producer, _, contract = archive_publication(frozen)
    bad_contract = copy.deepcopy(contract)
    bad_contract["frozen_inputs"]["reference_index"] = "0" * 64
    store = frozen["tmp"] / "registry"
    with pytest.raises(ValueError, match="frozen inputs differ"):
        catalog.register_producer(store, producer, origin="ARCHIVE", contract=bad_contract)
    with pytest.raises(ValueError, match="actual evaluated-run evidence"):
        catalog.register_producer(store, producer, origin="FRESH", contract=contract)
    assert not store.exists()


@pytest.mark.parametrize("protected", ["root", "run"])
def test_catalog_guards_archive_against_registry_and_receipt_writes(frozen, protected):
    catalog, producer, _, contract = archive_publication(frozen)
    before = snapshot(frozen["root"])
    destination = frozen[protected] / "must-not-be-created"
    with pytest.raises(ValueError, match="disjoint"):
        catalog.register_producer(destination, producer, origin="ARCHIVE", contract=contract)
    record = catalog.register_producer(frozen["tmp"] / "registry", producer, origin="ARCHIVE", contract=contract)
    with pytest.raises(ValueError, match="disjoint"):
        catalog.adopt(record, contract=contract, output=destination)
    assert not destination.exists()
    assert snapshot(frozen["root"]) == before


def test_catalog_guards_independent_thermal_archive_before_writing(frozen, thermal):
    catalog, producer, _, contract = archive_publication(frozen, thermal=thermal, endpoint="cts_thermal")
    before = snapshot(thermal["root"])
    destination = thermal["root"] / "must-not-be-created"
    with pytest.raises(ValueError, match="disjoint"):
        catalog.register_producer(destination, producer, origin="ARCHIVE", contract=contract)
    record = catalog.register_producer(frozen["tmp"] / "registry", producer, origin="ARCHIVE", contract=contract)
    with pytest.raises(ValueError, match="disjoint"):
        catalog.adopt(record, contract=contract, output=destination)
    assert not destination.exists()
    assert snapshot(thermal["root"]) == before
