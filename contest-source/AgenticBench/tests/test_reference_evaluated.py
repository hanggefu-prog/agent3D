"""Synthetic qualification fixtures: no EDA, Docker, or model execution."""
import copy
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from local_bench import reference_evaluated as evaluated
from open3dbench.backend.contracts import HBT_COLUMNS, NAME_MAP_COLUMNS
from test_one_pass_codesign_contract import endpoint_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def physical_checkpoint(root, stage, previous):
    """Schema-complete tiny physical checkpoint; no native database claims."""
    root.mkdir(parents=True, exist_ok=True)
    native = {"global_routing": "global_route", "detailed_routing": "route"}.get(stage, stage)
    def artifact(name, text="synthetic artifact\n"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text)
        return name
    cp = {"schema_version": 1, "stage": native, "pdk": "ng45"}
    if previous:
        cp["source_checkpoint"] = "../" + previous + "/checkpoint.json"
    if stage == "thermal":
        cp["thermal"] = {"algorithm": "hotspot-steady-grid", **{k: artifact(v) for k, v in {
            "metrics": "metrics.json", "power_trace": "power.ptrace", "steady": "output.steady",
            "grid_steady": "output.grid.steady", "layer_config": "layers.lcf"}.items()},
            "floorplans": {d: artifact(d + ".flp") for d in ("bottom", "top")}}
        write(root / "checkpoint.json", cp)
        return
    cp["dies"] = {d: {ext: artifact(d + "/design." + {"verilog": "v"}.get(ext, ext))
        for ext in ("odb", "def", "verilog", "sdc", "guide", "drc", "spef")} for d in ("bottom", "top")}
    cp["assembly"] = {k: artifact("assembly/" + k) for k in
        ("verilog", "sdc", "liberty", "timing_lef", "validation_tcl")}
    if stage not in {"placement", "pdn"}:
        cp["hbt_assignment"] = artifact("hbt.csv", ",".join(HBT_COLUMNS) + "\nH0,n,bottom,top,10,10,src,sink,B2T,0\n")
        cp["name_map"] = artifact("name_map.csv", ",".join(NAME_MAP_COLUMNS) + "\ninstance,x,bottom,x,x\n")
        cp["partition_assignment"] = artifact("partition_assignment.csv", "inst_name,die\nx,bottom\n")
        cp["timing_model"] = "timing_model.json"
        cp["timing_model_status"] = "provisional"
        write(root / "timing_model.json", {"status": "provisional"})
        cp["assembly"]["boundary"] = "assembly/boundary.json"
        write(root / "assembly/boundary.json", {"schema_version": 1,
            "units": {"resistance": "ohm", "capacitance": "pf", "time": "ns", "coordinate": "dbu"},
            "timing_model": {"status": "provisional", "model": "symmetric-lumped-rc-nldm",
                "resistance_ohm": 3., "capacitance_pf": .0006, "default_transition_ns": .01},
            "connections": [{"hbt_id": "H0", "source_die": "bottom", "sink_die": "top",
                             "direction": "B2T", "source_port": "src", "sink_port": "sink"}]})
    if stage == "hbt":
        cp["ready_for_opt"] = True
    if stage in {"opt", "cts", "global_routing", "detailed_routing", "report"}:
        algorithm = (json.loads((root / "metrics.json").read_text()).get("algorithm", "synthetic-native")
                     if (root / "metrics.json").exists() else "synthetic-native")
        if stage == "report":
            algorithm = "openrcx-continuous-passive-assembly-sta"
        cp[native] = {"algorithm": algorithm, "metrics": artifact("metrics.json", "{}")}
        cp[{"opt": "ready_for_cts", "cts": "ready_for_route", "global_routing": "ready_for_detailed_route",
            "detailed_routing": "ready_for_extract", "report": "ready_for_thermal"}[stage]] = True
    if stage == "report":
        cp["assembly"]["spefs"] = {d: artifact("assembly/" + d + ".spef") for d in ("bottom", "top")}
        cp["assembly"]["passive"] = {k: artifact("passive/" + k) for k in ("verilog", "spef", "plan", "validation_tcl")}
    write(root / "checkpoint.json", cp)


def seal_stages(run):
    for stage in json.loads((run / "run.json").read_text())["stages"]:
        root = run / evaluated.verifier.STAGE_DIRS[stage]
        target = root / "stage_manifest.json"
        manifest = json.loads(target.read_text())
        manifest["artifacts"] = {str(p.relative_to(root)): {"path": str(p.relative_to(root)),
            "bytes": p.stat().st_size, "sha256": evaluated.baseline.sha256(p)}
            for p in root.rglob("*") if p.is_file() and p != target}
        write(target, manifest)


@pytest.fixture
def make_evaluation(tmp_path, monkeypatch):
    # This suite validates contracts with synthetic runs. Do not require a
    # historical physical checkout or restore retired recipes into live source.
    source = tmp_path / "synthetic-producer"
    for relative in (evaluated.baseline.PARTITION_CONFIG, evaluated.baseline.PLACEMENT_CONFIG):
        write(source / relative, {"fixture": "reference-contract-test", "path": relative})
    monkeypatch.setattr(evaluated, "REPO", source)
    # Archive-prefix hashes have their own tests. All remaining production
    # provenance, artifact, checkpoint, seed, timing and thermal gates are real.
    monkeypatch.setattr(evaluated.baseline, "validate_physical_prefix", lambda run, case:
        evaluated.expected_frozen_inputs(case, "a" * 64)["physical_prefix_sha256"])
    def create(endpoint="timing", name="evaluation", *, rounds=5):
        output = tmp_path / name
        run = output / "run"
        profile, task_id, objective, stop = evaluated.ENDPOINTS[endpoint]
        from scripts.workspace_task_selection import workspace_task_id
        task_id = workspace_task_id(objective, rounds)
        endpoint_run(run, objective)
        if endpoint == "cts_thermal":
            path = run / "95_thermal_snapshot/metrics.json"
            data = json.loads(path.read_text())
            data["power"]["provenance"]["threads"] = 2
            write(path, data)
        image = sorted(evaluated.baseline.IMAGE_IDS)[0]
        physical = evaluated.expected_source_contract("bp_fe", endpoint, profile=profile, source_revision="1" * 40, image=image)
        config = copy.deepcopy(physical["configuration"])
        bindings = {}
        for domain, relative in (("partition", evaluated.baseline.PARTITION_CONFIG), ("placement", evaluated.baseline.PLACEMENT_CONFIG)):
            path = output / "source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((evaluated.REPO / relative).read_bytes())
            bindings[domain] = {"requested": config[domain]["config"], "origin": "explicit", "resolved": str(path),
                "source_relative": relative, "sha256": evaluated.baseline.sha256(path)}
            config[domain]["config"] = str(path)
        config_path = output / "evaluation-config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        (run / "config.yaml").write_text(yaml.safe_dump(evaluated._public_config(config_path)))
        task = evaluated.load_global_task(evaluated.FRAMEWORK / "global_tasks" / task_id)
        write(output / "evaluation-contract.json", {"schema_version": 1, "reference_run": None, "source_changes": [],
            "case": "bp_fe", "objective": objective, "opt_passes": 1, "task": task.to_dict(), "task_digest": task.digest(),
            "source_revision": "1" * 40, "image": image, "input_manifest_sha256": "a" * 64,
            "trusted_evaluator_patch_sha256": None, "profile_threads": 4,
            **({"thermal_snapshot_threads": 2} if endpoint == "cts_thermal" else {}),
            "evaluation_config_sha256": evaluated.baseline.sha256(config_path), "frontend_config_bindings": bindings,
            "evaluation_endpoint": {"profile": profile, "task_id": task_id, "backend_stop": stop, "opt_passes": 1}})
        write(run / "10_floorplan/io_metrics.json", {"seed": 1})
        write(run / "20_partition/partition_manifest.json", {"method": "tp-gnn-flat", "seed": 1001})
        write(run / "20_partition/tp_gnn_report.json", {"method": "tp-gnn-flat", "config": {"decoder": "global-2way"},
            "hierarchy": {"mode": "flat", "consumed_as_feature": False}})
        write(run / "30_placement/stages/02_global_place/stage_manifest.json", {"initialization_seed": 1000})
        if endpoint == "routed_thermal":
            write(run / "80_route/metrics.json", {"configuration": {"droute_seed": -1}})
        status = json.loads((run / "run.json").read_text())
        status.update(design="bp_fe", type="lol", pdk="ng45")
        write(run / "run.json", status)
        previous = None
        for stage in status["stages"]:
            root = run / evaluated.verifier.STAGE_DIRS[stage]
            root.mkdir(parents=True, exist_ok=True)
            if stage not in evaluated.verifier.TIMING_STAGES[:4]:
                physical_checkpoint(root, stage, previous)
                previous = root.name
            write(root / "fixture.json", {"fixture": stage})
            write(root / "stage_manifest.json", {"schema_version": 1, "stage": stage, "status": "success",
                "design": "bp_fe", "integration": "lol", "pdk": "ng45", "signature": {"fixture": "0" * 64},
                "method": {"partition": "tp-gnn-flat", "placement": "concurrent-dreamplace"}.get(stage, "native"),
                "timing": {"runtime_seconds": .5}, "provenance": {"repo_commit": "1" * 40,
                    "repo_dirty": False, "container_image_id": image}})
        seal_stages(run)
        return output, run, profile
    return create


@pytest.mark.parametrize("endpoint", evaluated.ENDPOINTS)
def test_independent_qualification_without_score_flags(make_evaluation, endpoint):
    output, run, profile = make_evaluation(endpoint)
    write(output / "verification.json", {"passed": False, "metrics": {"wns_setup_ns": 999999}})
    value = evaluated.prepare_evaluated_reference(run, "bp_fe", endpoint, profile=profile)
    assert value["qualified"] is True and value["executed_native_stages"] == []
    assert value["frozen_inputs"] == evaluated.expected_frozen_inputs("bp_fe", "a" * 64)
    assert value == evaluated.prepare_evaluated_reference(run, "bp_fe", endpoint, profile=profile)
    assert all(Path(row["path"]).is_relative_to(output) for row in value["evidence"])
    assert value["source_contract"] == evaluated.expected_source_contract("bp_fe", endpoint, profile=profile,
        source_revision="1" * 40, image=sorted(evaluated.baseline.IMAGE_IDS)[0])
    metric = "wns_setup_ns" if endpoint == "timing" else "global_average_temperature_k"
    assert value["baseline"]["raw_metrics"][metric] == (-1. if endpoint == "timing" else 322.)
    if endpoint == "cts_thermal":
        assert value["baseline"]["raw_metrics"]["boundary_budget_model"] == "passive-cutpoint"


@pytest.mark.parametrize("fault", ["missing_contract", "candidate", "changed_source", "wrong_case", "wrong_profile",
    "config_hash", "wrong_image", "task_digest", "task_definition", "input_digest", "source_revision", "unbound_config"])
def test_rejects_untrusted_evaluation_provenance(make_evaluation, fault):
    output, run, profile = make_evaluation()
    path = output / "evaluation-contract.json"
    value = json.loads(path.read_text())
    mutations = {"candidate": ("reference_run", "/a/reference/run"), "changed_source": ("source_changes", [{"path": "algo.py"}]),
        "wrong_case": ("case", "bp"), "config_hash": ("evaluation_config_sha256", "0" * 64),
        "wrong_image": ("image", "mutable:tag"), "task_digest": ("task_digest", "0" * 64),
        "input_digest": ("input_manifest_sha256", None), "source_revision": ("source_revision", "unknown")}
    if fault == "missing_contract":
        path.unlink()
    elif fault == "wrong_profile":
        value["evaluation_endpoint"]["profile"] = "tp_gnn_placement_two_pass"
    elif fault == "task_definition":
        value["task"]["objective"]["parameters"]["wns_weight"] = 0.
    elif fault == "unbound_config":
        value["frontend_config_bindings"]["placement"]["sha256"] = "0" * 64
    else:
        key, new = mutations[fault]
        value[key] = new
    if fault != "missing_contract":
        write(path, value)
    with pytest.raises(ValueError):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile=profile)


@pytest.mark.parametrize("fault", ["artifact_hash", "chain", "stage15", "producer", "other_pinned_image", "two_pass", "symlink", "bad_timing"])
def test_rejects_invalid_native_evidence(make_evaluation, fault):
    _, run, profile = make_evaluation()
    if fault == "artifact_hash":
        (run / "60_opt/metrics.json").write_text("{}")
    elif fault == "chain":
        path = run / "60_opt/checkpoint.json"
        value = json.loads(path.read_text()); value["source_checkpoint"] = "../40_pdn/checkpoint.json"
        write(path, value)
    elif fault == "stage15":
        path = run / "run.json"
        value = json.loads(path.read_text()); value["stages"].pop("timing_extract")
        write(path, value)
    elif fault == "producer":
        path = run / "60_opt/stage_manifest.json"
        value = json.loads(path.read_text()); value["provenance"]["repo_commit"] = "2" * 40
        write(path, value)
    elif fault == "other_pinned_image":
        path = run / "60_opt/stage_manifest.json"
        value = json.loads(path.read_text())
        value["provenance"]["container_image_id"] = next(image for image in evaluated.baseline.IMAGE_IDS
            if image != value["provenance"]["container_image_id"])
        write(path, value)
    elif fault == "symlink":
        path = run / "60_opt/fixture.json"
        path.unlink(); path.symlink_to(run / "50_hbt/fixture.json")
    else:
        path = run / "60_opt/metrics.json"
        value = json.loads(path.read_text())
        if fault == "two_pass":
            value["passes_run"] = 2
        else:
            value["timing_validity"]["missing_liberty_cells"] = 1
        write(path, value)
    if fault not in {"artifact_hash", "symlink"}:
        seal_stages(run)
    with pytest.raises(ValueError):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile=profile)


def test_plain_passed_flag_cannot_qualify(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    write(tmp_path / "verification.json", {"passed": True, "metrics": {"wns_setup_ns": 0}})
    with pytest.raises(ValueError, match="evidence"):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile="tp_gnn_placement_one_pass")


def test_ten_round_native_reference_reconstructs_the_registered_task(make_evaluation):
    from scripts.workspace_task_selection import load_registered_task, TIMING_TEN_TASK
    output, run, profile = make_evaluation(rounds=10)
    task = load_registered_task(TIMING_TEN_TASK, objective="timing")
    result = evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile=profile)
    assert result["source_identity"]["task_digest"] == task.digest()
    assert result["source_contract"]["opt_passes"] == 1
    recorded = json.loads((output / "evaluation-contract.json").read_text())
    assert recorded["task"]["budget"]["global_rounds"] == 10
    assert recorded["evaluation_endpoint"]["task_id"] == TIMING_TEN_TASK


@pytest.mark.parametrize("fault", ["old_digest", "old_definition", "arbitrary_id", "wrong_objective"])
def test_ten_round_reference_cannot_alias_another_task(make_evaluation, fault):
    output, run, profile = make_evaluation(rounds=10)
    path = output / "evaluation-contract.json"
    contract = json.loads(path.read_text())
    old = evaluated.load_global_task(evaluated.FRAMEWORK / "global_tasks/lol_partition_placement_timing_one_pass")
    if fault == "old_digest":
        contract["task_digest"] = old.digest()
    elif fault == "old_definition":
        contract["task"] = old.to_dict()
    elif fault == "arbitrary_id":
        contract["evaluation_endpoint"]["task_id"] = "../untrusted-task"
    else:
        contract["objective"] = "thermal"
    write(path, contract)
    with pytest.raises(ValueError):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile=profile)


@pytest.mark.parametrize("fault", ["power_model", "power_checkpoint", "power_threads", "contract_threads", "mean", "warnings", "run_config", "snapshot_symlink"])
def test_cts_thermal_provenance_and_configuration_are_not_score_flags(make_evaluation, fault):
    output, run, profile = make_evaluation("cts_thermal")
    path = run / "95_thermal_snapshot/metrics.json"
    data = json.loads(path.read_text())
    if fault == "power_model":
        data["power"]["activity_model"] = "candidate-invented"
    elif fault == "power_checkpoint":
        data["power"]["provenance"]["checkpoint"] = "/another/run/70_cts/checkpoint.json"
    elif fault == "power_threads":
        data["power"]["provenance"]["threads"] = 32
    elif fault == "contract_threads":
        receipt = json.loads((output / "evaluation-contract.json").read_text())
        receipt["thermal_snapshot_threads"] = 32
        write(output / "evaluation-contract.json", receipt)
    elif fault == "mean":
        data["temperature"]["global_average_k"] = 0.
    elif fault == "warnings":
        data["power_extraction"]["top"]["warning_count"] = 1
    elif fault == "run_config":
        config = yaml.safe_load((run / "config.yaml").read_text())
        config["placement"]["target_density"] = .7
        (run / "config.yaml").write_text(yaml.safe_dump(config))
    else:
        (run / "95_thermal_snapshot/alias").symlink_to(output / "evaluation-contract.json")
    write(path, data)
    with pytest.raises(ValueError):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "cts_thermal", profile=profile)


@pytest.mark.parametrize("fault", [None, "bytes", "manifest", "candidate", "purpose", "inventory", "record"])
def test_extend_requires_verified_parent_and_unchanged_native_prefix(make_evaluation, tmp_path, fault):
    _, parent_run, profile = make_evaluation("timing", "parent")
    parent = evaluated.prepare_evaluated_reference(parent_run, "bp_fe", "timing", profile=profile)
    record = tmp_path / "parent-record.json"
    write(record, {"synthetic_registry_record": True})
    parent.update(reference_id="b" * 64, record=str(record), record_sha256=evaluated.baseline.sha256(record))
    output, run, profile = make_evaluation("cts_thermal", "extended")
    inventory = []
    for stage in evaluated.verifier.TIMING_STAGES:
        directory = evaluated.verifier.STAGE_DIRS[stage]
        shutil.rmtree(run / directory)
        shutil.copytree(parent_run / directory, run / directory)
        inventory.append({"stage": stage, "manifest_sha256": evaluated.baseline.sha256(run / directory / "stage_manifest.json")})
    path = output / "evaluation-contract.json"
    contract = json.loads(path.read_text())
    contract.update(reference_purpose="EXTEND", parent_reference_id=parent["reference_id"], reused_prefix=inventory)
    if fault == "bytes":
        (run / "60_opt/fixture.json").write_text("changed")
    elif fault == "manifest":
        write(run / "60_opt/stage_manifest.json", {"passed": True})
    elif fault == "candidate":
        contract["reference_run"] = str(parent_run)
    elif fault == "purpose":
        contract["reference_purpose"] = "candidate"
    elif fault == "inventory":
        contract["reused_prefix"].pop(2)
    elif fault == "record":
        record.write_text("changed")
    write(path, contract)
    if fault:
        with pytest.raises(ValueError):
            evaluated.prepare_evaluated_reference(run, "bp_fe", "cts_thermal", profile=profile, parent_reference=parent)
    else:
        result = evaluated.prepare_evaluated_reference(run, "bp_fe", "cts_thermal", profile=profile, parent_reference=parent)
        assert result["qualified"] is True and result["source_identity"]["parent_reference_id"] == "b" * 64
        assert result["executed_native_stages"] == []


def test_fresh_then_extend_publications_are_read_only_adoptable(make_evaluation, tmp_path):
    from local_bench import reference_catalog as catalog
    from local_bench.reference_store import file_sha, load_reference
    store = tmp_path / "registry"
    output, run, profile = make_evaluation("timing", "original")
    producer = {"kind": "native-evaluation-v1", "run_root": str(run), "case": "bp_fe",
                "endpoint": "timing", "profile": profile, "evaluation_root": str(output)}
    parent_payload = catalog.qualify_producer(producer)
    parent_contract = catalog.make_contract(parent_payload, task_digest="a" * 64, evaluator_digest="c" * 64)
    record = catalog.register_producer(store, producer, origin="FRESH", contract=parent_contract)
    parent = load_reference(record, contract=parent_contract, validate=catalog.qualify_producer)
    assert parent["origin"] == "FRESH"
    old_run = run
    output, run, profile = make_evaluation("cts_thermal", "suffix")
    inventory = []
    for stage in evaluated.verifier.TIMING_STAGES:
        directory = evaluated.verifier.STAGE_DIRS[stage]
        shutil.rmtree(run / directory)
        shutil.copytree(old_run / directory, run / directory)
        inventory.append({"stage": stage, "manifest_sha256": file_sha(run / directory / "stage_manifest.json")})
    path = output / "evaluation-contract.json"
    contract = json.loads(path.read_text())
    contract.update(reference_purpose="EXTEND", parent_reference_id=parent["reference_id"], reused_prefix=inventory)
    write(path, contract)
    producer = {"kind": "native-evaluation-v1", "run_root": str(run), "case": "bp_fe", "endpoint": "cts_thermal",
        "profile": profile, "evaluation_root": str(output), "parent_record": str(record),
        "parent_reference_id": parent["reference_id"]}
    payload = catalog.qualify_producer(producer)
    binding = catalog.make_contract(payload, task_digest="d" * 64, evaluator_digest="c" * 64)
    extended = catalog.register_producer(store, producer, origin="EXTEND", contract=binding)
    adopted = load_reference(extended, contract=binding, validate=catalog.qualify_producer)
    assert adopted["origin"] == "EXTEND"
    assert adopted["qualification"] == payload
    assert payload["executed_native_stages"] == []
    # Parent mutation invalidates the child even when the child's own files stay intact.
    (old_run / "60_opt/fixture.json").write_text("tampered parent")
    with pytest.raises(ValueError):
        load_reference(extended, contract=binding, validate=catalog.qualify_producer)


def add_native_opt_log_aliases(run):
    """Match native OPT's final-copy layout, leaving artifacts as real files."""
    root = run / "60_opt"
    pairs = [(f"{die}/{name}.log", f"iterations/01/{die}/{name}.log")
             for die in ("bottom", "top") for name in ("repair_design", "repair_timing")]
    pairs += [(f"{die}/{kind}/{view}/proof.log", f"iterations/01/{die}/{kind}/{view}/proof.log")
              for die in ("bottom", "top")
              for kind in ("sdc_evidence", "timing_sdc_evidence", "final_sdc_evidence")
              for view in ("bottom", "top")]
    pairs += [("assembly/passive/sta.log", "iterations/01/global/passive/sta.log")]
    for relative, target in pairs:
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic native log: " + target + "\n")
    seal_stages(run)
    aliases = []
    for relative, target in pairs:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(root / target, path.parent))
        aliases.append(path)
    return aliases


@pytest.mark.parametrize("endpoint", evaluated.ENDPOINTS)
def test_exact_native_opt_log_aliases_are_qualified_read_only(make_evaluation, endpoint):
    _, run, profile = make_evaluation(endpoint)
    aliases = add_native_opt_log_aliases(run)
    before = {str(p): (os.readlink(p), p.lstat()) for p in aliases}
    value = evaluated.prepare_evaluated_reference(run, "bp_fe", endpoint, profile=profile)
    assert value["qualified"] and value["executed_native_stages"] == []
    assert len(value["native_log_aliases"]) == len(aliases) == 17
    assert len(evaluated.OPT_FINAL_LOG_ALIASES) == 17
    evidence = {row["path"]: row for row in value["evidence"]}
    assert not set(map(str, aliases)) & set(evidence)
    for alias in value["native_log_aliases"]:
        path, target = Path(alias["path"]), Path(alias["target_path"])
        assert alias["link_text"] == os.readlink(path)
        assert not Path(alias["link_text"]).is_absolute()
        assert target == path.resolve() and not target.is_symlink()
        assert alias["target_bytes"] == evidence[str(target)]["bytes"] == target.stat().st_size
        assert alias["target_sha256"] == evidence[str(target)]["sha256"] == evaluated.baseline.sha256(target)
        assert alias["link_identity"]["inode"] == path.lstat().st_ino
    assert before == {str(p): (os.readlink(p), p.lstat()) for p in aliases}
    assert value == evaluated.prepare_evaluated_reference(run, "bp_fe", endpoint, profile=profile)


@pytest.mark.parametrize("fault", ["absolute", "pass02", "wrong_die", "other_stage", "external", "noncanonical",
    "dangling", "target_link", "target_parent_link", "alias_parent_link", "unknown_log", "directory",
    "metrics", "database", "checkpoint", "manifest", "manifest_lists_alias"])
def test_native_opt_alias_exception_is_fail_closed(make_evaluation, tmp_path, fault):
    _, run, profile = make_evaluation()
    aliases = add_native_opt_log_aliases(run)
    link = aliases[0]
    target = link.resolve()
    if fault in {"absolute", "pass02", "wrong_die", "other_stage", "external", "noncanonical"}:
        replacement = {"absolute": str(target), "pass02": "../iterations/02/bottom/repair_design.log",
            "wrong_die": "../iterations/01/top/repair_design.log", "other_stage": "../../50_hbt/fixture.json",
            "external": str(tmp_path / "outside.log"),
            "noncanonical": "../iterations/01/../01/bottom/repair_design.log"}[fault]
        link.unlink(); link.symlink_to(replacement)
    elif fault == "dangling":
        target.unlink()
    elif fault == "target_link":
        target.unlink(); target.symlink_to(run / "50_hbt/fixture.json")
    elif fault == "target_parent_link":
        parent = target.parent
        moved = parent.with_name("hidden-bottom")
        parent.rename(moved); parent.symlink_to(moved, target_is_directory=True)
    elif fault == "alias_parent_link":
        parent = link.parent
        moved = parent.with_name("hidden-bottom")
        parent.rename(moved); parent.symlink_to(moved, target_is_directory=True)
    elif fault in {"unknown_log", "directory"}:
        (run / "60_opt" / ("unrecognized.log" if fault == "unknown_log" else "shortcut")).symlink_to(
            target if fault == "unknown_log" else target.parent)
    elif fault == "manifest_lists_alias":
        path = run / "60_opt/stage_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["artifacts"]["untrusted_alias"] = {"path": str(link.relative_to(path.parent)),
            "bytes": target.stat().st_size, "sha256": evaluated.baseline.sha256(target)}
        write(path, manifest)
    else:
        path = run / "60_opt" / {"metrics": "metrics.json", "database": "bottom/design.odb",
            "checkpoint": "checkpoint.json", "manifest": "stage_manifest.json"}[fault]
        moved = path.with_name("real-" + path.name)
        path.rename(moved); path.symlink_to(moved.name)
    with pytest.raises(ValueError):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile=profile)


@pytest.mark.parametrize("fault", ["link_text", "link_identity", "target_bytes", "target_identity"])
def test_opt_alias_rejects_changes_during_its_read(make_evaluation, monkeypatch, fault):
    _, run, _ = make_evaluation()
    link = add_native_opt_log_aliases(run)[0]
    target = link.resolve()
    original = evaluated._record
    def changed(path, role):
        record = original(path, role)
        if path == target:
            if fault.startswith("link"):
                text = os.readlink(link)
                # Rename retains the old inode, preventing accidental inode reuse.
                link.rename(link.with_name("previous-link"))
                link.symlink_to(text if fault == "link_identity" else "../iterations/01/top/repair_design.log")
            elif fault == "target_bytes":
                target.write_text("changed target\n")
            else:
                target.rename(target.with_name("previous-target"))
                target.write_text("Synthetic native log: iterations/01/bottom/repair_design.log\n")
        return record
    monkeypatch.setattr(evaluated, "_record", changed)
    with pytest.raises(ValueError, match="changed during qualification"):
        evaluated._observe_opt_log_alias(run, link)


def test_opt_alias_rechecked_after_other_evidence_reads(make_evaluation, monkeypatch):
    _, run, profile = make_evaluation()
    link = add_native_opt_log_aliases(run)[0]
    original = evaluated.verifier._timing_row
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        text = os.readlink(link)
        link.rename(link.with_name("previous-link")); link.symlink_to(text)
        return result
    monkeypatch.setattr(evaluated.verifier, "_timing_row", changed)
    with pytest.raises(ValueError, match="changed during qualification"):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile=profile)


def test_opt_alias_inventory_rechecked_after_other_reads(make_evaluation, monkeypatch):
    _, run, profile = make_evaluation()
    target = add_native_opt_log_aliases(run)[0].resolve()
    original = evaluated.verifier._timing_row
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (run / "60_opt/unexpected.log").symlink_to(target)
        return result
    monkeypatch.setattr(evaluated.verifier, "_timing_row", changed)
    with pytest.raises(ValueError, match="inventory changed during qualification"):
        evaluated.prepare_evaluated_reference(run, "bp_fe", "timing", profile=profile)


@pytest.mark.parametrize("fault", [None, "link_text", "link_identity", "target", "remove_link", "regular_link"])
def test_registered_opt_aliases_are_rechecked_on_adopt(make_evaluation, tmp_path, fault):
    from local_bench import reference_catalog as catalog
    from local_bench.reference_store import adopt_reference
    output, run, profile = make_evaluation(rounds=10)
    aliases = add_native_opt_log_aliases(run)
    producer = {"kind": "native-evaluation-v1", "run_root": str(run), "case": "bp_fe",
                "endpoint": "timing", "profile": profile, "evaluation_root": str(output)}
    payload = catalog.qualify_producer(producer)
    contract = catalog.make_contract(payload, task_digest="a" * 64, evaluator_digest="c" * 64)
    record = catalog.register_producer(tmp_path / "registry", producer, origin="FRESH", contract=contract)
    link = aliases[0]
    text, target = os.readlink(link), link.resolve()
    if fault in {"link_text", "link_identity"}:
        link.rename(link.with_name("previous-link"))
        link.symlink_to(text if fault == "link_identity" else "../iterations/01/top/repair_design.log")
        link.with_name("previous-link").unlink()
    elif fault == "target":
        target.write_text("tampered native target\n")
    elif fault in {"remove_link", "regular_link"}:
        link.unlink()
        if fault == "regular_link":
            link.write_bytes(target.read_bytes())
    if fault:
        with pytest.raises(ValueError):
            adopt_reference(record, contract=contract, validate=catalog.qualify_producer, output=tmp_path / "adopt")
    else:
        receipt = adopt_reference(record, contract=contract, validate=catalog.qualify_producer, output=tmp_path / "adopt")
        assert receipt["status"] == "qualified" and receipt["new_reference_native_attempts"] == 0
        assert receipt["model_calls"] == receipt["candidate_finals"] == 0
        assert receipt["qualification"]["native_log_aliases"] == payload["native_log_aliases"]
