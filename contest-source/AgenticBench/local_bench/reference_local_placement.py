"""Same-runtime native reference producer for the two placement smoke tasks.

Qualification is read-only. Historical config equality is deliberately not a
source/hook compatibility bridge. Only controller-declared FRESH initial runs
with a measured installed-source audit can become future ADOPT references.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import subprocess

import yaml

from . import reference_baseline as baseline
from .benchmark_input import validate_benchmark_input
from .model import load_candidate, load_task, resolve_task_path
from .program_bridge import initial_program_text, materialize_program
from .reference_placement_evidence import DYNAMIC_TASK, SOURCE_TASK, STAGE, TASKS, validate_reference_placement_evidence
from .reference_store import digest, file_sha

FRAMEWORK = Path(__file__).resolve().parents[1]
SOURCE = FRAMEWORK.parent
DREAMPLACE = Path("Place-3D/thirdparty/DREAMPlace/dreamplace")
RUNTIME_MODULES = ("Placer.py", "Params.py", "PlaceDB.py", "BasicPlace.py", "PlaceObj.py",
                   "NonLinearPlace.py", "NesterovAcceleratedGradientOptimizer.py")
ENDPOINTS = {SOURCE_TASK: "placement_source", DYNAMIC_TASK: "placement_dynamic"}


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Missing canonical regular placement reference evidence: " + str(path))
    return path


def _json(path):
    value = json.loads(_regular(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Placement reference evidence must be a JSON object")
    # Reject NaN/Infinity, including unused producer fields.
    digest(value)
    return value


def _source_revision(source, *, clean=False):
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       text=True, env=env).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Local reference source needs a full commit identity")
    if clean and subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
            "--untracked-files=all", "--ignore-submodules=none"], text=True, env=env).strip():
        raise ValueError("Local reference selection requires the exact clean initial source checkout")
    return revision


def _context(source_root, task_id, case, benchmark_input):
    from .reference_latest import baseline_for_bundle, BASELINE_ID
    source = Path(source_root).resolve(strict=True)
    if task_id not in TASKS:
        raise ValueError("Unsupported native local placement reference task")
    spec = load_task(resolve_task_path(FRAMEWORK / "local_tasks", task_id))
    bundle = Path(benchmark_input).resolve(strict=True)
    selected_baseline = baseline_for_bundle(bundle)
    if source != SOURCE and selected_baseline.BASELINE_ID != BASELINE_ID:
        raise ValueError("Local reference source must be the trusted controller checkout")
    validated = validate_benchmark_input(bundle, task_id=task_id)
    aligned = selected_baseline.validate_reference_bundle(bundle, case, fixed_partition=True)
    config = Path(validated["config"]).resolve(strict=True)
    recipe = yaml.safe_load(config.read_text())
    selected_baseline.validate_reference_config(recipe, case)
    spec = replace(spec, base_config=config)
    initial = initial_program_text(spec, load_candidate(spec.initial_candidate))
    candidate = materialize_program(spec, initial, source)
    manifest = bundle / "benchmark_input.json" if bundle.is_dir() else bundle
    frozen = {"benchmark_input_manifest_sha256": file_sha(_regular(manifest)),
              "physical_prefix_sha256": aligned["physical_input_sha256"]}
    return source, spec, candidate, recipe, frozen, Path(validated["run_dir"]), manifest


def _physical(source, spec, candidate, recipe, source_revision, image):
    from .reference_latest import baseline_for_config, BASELINE_ID, SOURCE_REVISION
    family = baseline_for_config(recipe)
    family.validate_reference_image(image)
    if family.BASELINE_ID == BASELINE_ID and source_revision != SOURCE_REVISION:
        raise ValueError("Latest local reference source revision differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_revision)):
        raise ValueError("Local reference source needs a full commit identity")
    # Include the task-owned hook materializer and the native command owner.
    files = [spec.root / "task.yaml", spec.program_template, spec.program_adapter,
             source / "Place-3D/src/placement/concurrent_dreamplace.py",
             FRAMEWORK / "local_bench/evaluation.py", FRAMEWORK / "local_bench/workspace.py",
             FRAMEWORK / "local_bench/program_bridge.py", FRAMEWORK / "scripts/local_reference.py",
             FRAMEWORK / "scripts/run_local_task_smoke.py", FRAMEWORK / "scripts/codesign_runtime.py"]
    if spec.task_id == DYNAMIC_TASK:
        files.append(spec.root / "nesterov_runtime.py")
    return {"producer_contract": "local-placement-same-runtime-v1", "task_id": spec.task_id,
            "source_revision": source_revision, "image": image,
            "configuration": recipe, "physical_config_sha256": digest(recipe),
            "initial_candidate_sha256": digest(candidate),
            "runtime_python_sha256": {name: file_sha(_regular(source / DREAMPLACE / name))
                                      for name in RUNTIME_MODULES},
            "implementation_sha256": {_implementation_name(source, path): file_sha(_regular(path))
                                      for path in files},
            "runtime_loading": "installed-dreamplace-with-verified-initial-overlay-v1",
            "evaluation_boundary": "dreamplace-global-placement", "threads": 4,
            "archive_source_bridge": "unsupported"}


def _implementation_name(source, path):
    try:
        return "AgenticBench/" + str(path.relative_to(FRAMEWORK))
    except ValueError:
        return str(path.relative_to(source))


def _implementation_path(source, name):
    if name.startswith("AgenticBench/"):
        return FRAMEWORK / name[len("AgenticBench/"):]
    return source / name


def expected_local_contract(source_root, task_id, case, benchmark_input, *, source_revision, image):
    """Independently construct the consumer identity before any reference lookup."""
    from .reference_catalog import evaluator_digest, task_digest
    source, spec, candidate, recipe, frozen, _, _ = _context(source_root, task_id, case, benchmark_input)
    if _source_revision(source, clean=True) != source_revision:
        raise ValueError("Local reference source revision differs from the controller")
    return {"case": case, "endpoint": ENDPOINTS[task_id], "task_digest": task_digest(spec.root),
            "evaluator_digest": evaluator_digest(), "frozen_inputs": frozen,
            "physical": _physical(source, spec, candidate, recipe, source_revision, image)}


def audit_installed_runtime(receipt_path, output_path):
    """Container-only preflight: measure Python source before any native launch.

    Invoked by the trusted transport, never by qualification or discovery. It
    does not import DREAMPlace, touch model APIs, or dispatch a physical tool.
    """
    receipt = _json(receipt_path)
    if receipt.get("reference_purpose") != "FRESH":
        raise ValueError("Runtime audit requires a controller FRESH receipt")
    physical = receipt["contract"]["physical"]
    image = os.environ.get("OPEN3DBENCH_IMAGE_ID")
    if image != physical["image"]:
        raise ValueError("Installed runtime audit image differs from the pinned contract")
    measured = {name: file_sha(_regular(Path("/opt/dreamplace/dreamplace") / name))
                for name in RUNTIME_MODULES}
    if measured != physical["runtime_python_sha256"]:
        raise ValueError("Installed DREAMPlace source/hook compatibility is unqualified; no physical flow dispatched")
    value = {"schema_version": 1, "image": image, "runtime_root": "/opt/dreamplace/dreamplace",
             "reference_purpose_sha256": file_sha(receipt_path), "python_sha256": measured}
    with Path(output_path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _expected_overlay(source, candidate):
    """Apply the trusted initial adapter's diff in memory, never run candidate code."""
    patch = candidate.get("source_patch", "")
    outputs = {}
    lines = patch.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- a/") or index + 1 >= len(lines):
            raise ValueError("Malformed trusted initial source patch")
        name = lines[index][6:].rstrip("\n")
        if lines[index + 1] != "+++ b/" + name + "\n" or name in outputs:
            raise ValueError("Ambiguous trusted initial source patch")
        original = _regular(source / name).read_text().splitlines(keepends=True)
        result, cursor = [], 0
        index += 2
        while index < len(lines) and lines[index].startswith("@@"):
            match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?", lines[index])
            if not match:
                raise ValueError("Malformed trusted initial source hunk")
            start = int(match[1]) - 1
            if start < cursor:
                raise ValueError("Overlapping trusted source hunks")
            result.extend(original[cursor:start])
            cursor = start
            index += 1
            while index < len(lines) and not lines[index].startswith(("@@", "--- a/")):
                line = lines[index]
                if line[0] in " -":
                    if cursor >= len(original) or original[cursor] != line[1:]:
                        raise ValueError("Trusted initial patch source differs")
                    cursor += 1
                if line[0] in " +":
                    result.append(line[1:])
                if line[0] not in " +-":
                    raise ValueError("Unsupported trusted source hunk")
                index += 1
        result.extend(original[cursor:])
        outputs[name] = "".join(result)
    return outputs


def _placement_def(path, partition):
    """Check native placement completeness without asserting backend legality."""
    with _regular(partition).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row.get("inst_name") for row in rows]
    if not names or None in names or len(set(names)) != len(names):
        raise ValueError("Invalid fixed placement instance inventory")
    text = _regular(path).read_text()
    match = re.search(r"\bCOMPONENTS\s+(\d+)\s*;(.*?)\bEND\s+COMPONENTS\b", text, re.S)
    if not match:
        raise ValueError("Native placement DEF lacks components")
    entries = re.findall(r"-\s+(\S+)\s+\S+(.*?);", match[2], re.S)
    placed = {}
    for name, tail in entries:
        if name in placed or not re.search(r"\+\s+(?:PLACED|FIXED)\s*\(\s*-?\d+\s+-?\d+\s*\)\s+\S+", tail):
            raise ValueError("Native placement DEF has duplicate or unplaced components")
        placed[name] = True
    if len(entries) != int(match[1]) or not set(names).issubset(placed):
        raise ValueError("Native placement DEF is incomplete for the fixed partition")


def _strict_native_metrics(run):
    from .metrics import _placement_native
    final = None
    for line in (run / STAGE / "run.log").read_text().splitlines():
        if not all(label in line for label in ("wHPWL", "Overflow", "MaxDensity")):
            continue
        matches = [re.search(pattern, line) for pattern in
                   (r"wHPWL\s+([^,\s]+)", r"(?<!Global )Overflow\s*\[([^\]]+)\]", r"MaxDensity\s*\[([^\]]+)\]")]
        if not all(matches):
            raise ValueError("Native placement lacks a complete two-die metric row")
        try:
            hpwl = float(matches[0][1])
            overflow, density = ([float(v) for v in match[1].replace(",", " ").split()]
                                 for match in matches[1:])
        except ValueError as error:
            raise ValueError("Malformed native placement metric row") from error
        final = (hpwl, overflow, density)
    if final is None or len(final[1]) != 2 or len(final[2]) != 2:
        raise ValueError("Missing native two-die placement metrics")
    if any(not math.isfinite(v) or v < 0 for v in [final[0], *final[1], *final[2]]):
        raise ValueError("Native placement metrics must be finite and nonnegative")
    raw = _placement_native(run)
    expected = {"placement_native_hpwl": final[0], "placement_overflow_bottom": final[1][0],
                "placement_overflow_top": final[1][1], "placement_max_density_bottom": final[2][0],
                "placement_max_density_top": final[2][1]}
    if any(raw.get(k) != v for k, v in expected.items()):
        raise ValueError("Native placement scorer and independent metrics disagree")
    return raw


def _validate_copied_input_params(baseline, run, relative, frozen):
    """Allow exactly the copier's three path substitutions, never physical edits."""
    if relative not in {"00_input/design_params.json", "10_floorplan/design_params.json"}:
        raise ValueError("Unsupported copied parameter artifact")
    expected = _json(baseline / relative)
    for key, name in (("def_input", "design.def"), ("verilog_input", "design.v"), ("sdc_input", "design.sdc")):
        artifact = str(Path(relative).parent / name)
        if (run / artifact).is_file():
            sha = frozen.get(artifact)
            if not sha or any(file_sha(_regular(root / artifact)) != sha for root in (baseline, run)):
                raise ValueError("Rebased placement input is not bound to immutable physical bytes: " + artifact)
            expected[key] = str(_regular(run / artifact))
    if _json(run / relative) != expected:
        raise ValueError("Copied placement parameters changed beyond the three input paths")


def prepare_local_placement_reference(evaluation_root, task_id, case, *, source_root, benchmark_input):
    """Independently qualify an actual FRESH initial run; never cached scores."""
    from .evaluation import _dreamplace_global_placement_command, _validate_dreamplace_global_output
    from .placement_baseline import validate_tp_gnn_partition
    from .reference_catalog import evaluator_digest, task_digest
    evaluation = Path(evaluation_root).resolve(strict=True)
    source, spec, candidate, recipe, frozen, baseline_run, manifest_path = _context(
        source_root, task_id, case, benchmark_input)
    from .reference_latest import baseline_for_config, BASELINE_ID, bind_recipe_paths
    latest_family = baseline_for_config(recipe).BASELINE_ID == BASELINE_ID
    parent, run = evaluation.parent, evaluation / "run"
    workspace = evaluation / "workspace/Open3DBench-V2.0"
    receipt_path = parent / "reference-purpose.json"
    receipt = _json(receipt_path)
    ledger_path = parent / (evaluation.name + ".attempt.json")
    ledger = _json(ledger_path)
    contract = receipt.get("contract", {})
    physical = _physical(source, spec, candidate, recipe, _source_revision(source, clean=True), receipt.get("image"))
    expected = {"case": case, "endpoint": ENDPOINTS[task_id], "task_digest": task_digest(spec.root),
                "evaluator_digest": evaluator_digest(), "frozen_inputs": frozen, "physical": physical}
    if (receipt.get("schema_version") != 1 or receipt.get("reference_purpose") != "FRESH"
            or receipt.get("reference_run", "missing") is not None or contract != expected
            or receipt.get("evaluation_root") != str(evaluation)
            or receipt.get("source_root") != str(source)):
        raise ValueError("Local reference requires exact controller FRESH provenance; candidates cannot become references")
    candidate_path = parent / "candidate/candidate.json"
    if _json(candidate_path) != candidate or receipt.get("candidate_sha256") != file_sha(candidate_path):
        raise ValueError("Local reference must be the unmodified task initial program")
    execution_contract = ledger.get("contract", {})
    required = {"kind": "local-task-smoke-evaluation", "task": task_id,
                "candidate_sha256": file_sha(candidate_path), "source_commit": physical["source_revision"],
                "image": physical["image"], "reference": None, "create_routing_reference": False,
                "reference_purpose": "FRESH", "reference_purpose_sha256": file_sha(receipt_path)}
    if (ledger.get("schema_version") != 2 or ledger.get("state") != "exited"
            or ledger.get("exit_code") != 0 or ledger.get("evaluator_process_started") is not True
            or any(execution_contract.get(key) != value for key, value in required.items())):
        raise ValueError("Local reference lacks successful controller-owned FRESH execution provenance")
    audit_path = parent / "runtime-source.json"
    audit = _json(audit_path)
    if audit != {"schema_version": 1, "image": physical["image"],
                 "runtime_root": "/opt/dreamplace/dreamplace", "reference_purpose_sha256": file_sha(receipt_path),
                 "python_sha256": physical["runtime_python_sha256"]}:
        raise ValueError("Installed runtime source/hook compatibility audit differs")
    evidence = {}
    def add(path, role):
        path = _regular(path)
        evidence[str(path)] = {"path": str(path), "bytes": path.stat().st_size,
                               "sha256": file_sha(path), "role": role}
    for path, role in ((receipt_path, "reference-purpose"), (ledger_path, "controller-execution"),
                       (audit_path, "installed-source-audit"), (candidate_path, "initial-program"),
                       (manifest_path, "frozen-input-manifest")):
        add(path, role)
    for relative, expected_hash in frozen["physical_prefix_sha256"].items():
        for root in (baseline_run, run):
            path = _regular(root / relative)
            if latest_family and root == run and relative in {
                    "00_input/design_params.json", "10_floorplan/design_params.json"}:
                _validate_copied_input_params(baseline_run, run, relative, frozen["physical_prefix_sha256"])
                add(path, "path-rebased-immutable-placement-input")
                continue
            if file_sha(path) != expected_hash:
                raise ValueError("Local reference frozen prefix changed: " + relative)
            add(path, "immutable-placement-input")
    validate_tp_gnn_partition(run, expected_method=recipe["partition"]["algorithm"], require_bound_report=True)
    for name in ("stage_manifest.json", "partition_manifest.json", "tp_gnn_report.json"):
        add(run / "20_partition" / name, "bound-partition-evidence")
    result_path = evaluation / "evaluation_result.json"
    result = _json(result_path)
    artifacts = result.get("artifacts", {})
    if (artifacts.get("task") != task_id or artifacts.get("run_dir") != str(run)
            or artifacts.get("workspace") != str(workspace)
            or artifacts.get("candidate") != str(candidate_path.parent)
            or artifacts.get("output_dir") != str(evaluation) or artifacts.get("exit_code") != 0):
        raise ValueError("Native placement evaluation paths/task differ")
    add(result_path, "native-process-provenance")
    config_path = evaluation / "resolved_config.yaml"
    config = yaml.safe_load(_regular(config_path).read_text())
    expected_config = json.loads(json.dumps(recipe))
    if latest_family:
        expected_config = bind_recipe_paths(expected_config, source)
    expected_config["placement"].update(candidate["config_overrides"]["placement"])
    expected_config["placement"]["config"] = str(evaluation / "candidate_files/placement.json")
    if config != expected_config:
        raise ValueError("Native placement resolved recipe differs")
    add(config_path, "resolved-physical-configuration")
    flow_path = evaluation / "candidate_files/placement.json"
    expected_flow = json.loads(json.dumps(candidate["candidate_files"]["placement.json"]))
    expected_flow["concurrent_dreamplace"]["initialization_seed"] = 1000
    if _json(flow_path) != expected_flow:
        raise ValueError("Native placement initial flow config differs")
    add(flow_path, "initial-placement-configuration")
    params_input = evaluation / "harness_inputs/design_params.json"
    if _json(params_input) != _json(run / "10_floorplan/design_params.json"):
        raise ValueError("Native placement base parameters differ from frozen input")
    add(params_input, "frozen-runtime-input")
    command, cwd = _dreamplace_global_placement_command(spec, config, workspace, run,
        "/usr/bin/python3", base_params_override=params_input,
        reference_family=BASELINE_ID if latest_family else None)
    invocations = artifacts.get("native_flow_execution")
    if not isinstance(invocations, list) or len(invocations) != 1:
        raise ValueError("Local reference requires exactly one physical placement invocation")
    invocation = invocations[0]
    if (artifacts.get("command") != command or invocation.get("command") != command
            or invocation.get("cwd") != str(cwd) or invocation.get("process_started") is not True
            or invocation.get("returncode") != 0 or invocation.get("status") != "completed"):
        raise ValueError("Local reference native command was not successfully executed")
    seconds = invocation.get("wall_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Local reference native runtime evidence is missing")
    native = _validate_dreamplace_global_output(run)
    for path in native.values():
        add(path, "native-placement")
    stage_manifest = _json(run / STAGE / "stage_manifest.json")
    overlay = evaluation / "runtime/dreamplace-install"
    root = overlay / "dreamplace" if task_id == DYNAMIC_TASK else Path("/opt/dreamplace/dreamplace")
    if (stage_manifest.get("stage") != "02_global_place" or stage_manifest.get("stage_dir") != str(run / STAGE)
            or stage_manifest.get("dreamplace_root") != str(root)
            or stage_manifest.get("partition_dir") != str(run / "20_partition")
            or stage_manifest.get("params") != str(run / STAGE / "params.json")
            or stage_manifest.get("log") != str(run / STAGE / "run.log")
            or stage_manifest.get("command") != ["/usr/bin/python3", str(root / "Placer.py"), str(run / STAGE / "params.json")]):
        raise ValueError("Native placement did not load the qualified runtime/hook")
    patched = _expected_overlay(source, candidate)
    if task_id == SOURCE_TASK and (patched or artifacts.get("dreamplace_runtime") is not None):
        raise ValueError("Source initial reference unexpectedly changed native code")
    if task_id == DYNAMIC_TASK:
        if artifacts.get("dreamplace_runtime") != str(overlay):
            raise ValueError("Dynamic initial reference lacks its active runtime overlay")
        for name in RUNTIME_MODULES:
            relative = str(DREAMPLACE / name)
            expected_text = patched.get(relative, (source / relative).read_text())
            path = _regular(overlay / "dreamplace" / name)
            if path.read_text() != expected_text:
                raise ValueError("Dynamic reference loaded different runtime source: " + name)
            add(path, "active-runtime-source")
    wrapper = workspace / "Place-3D/src/placement/concurrent_dreamplace.py"
    if file_sha(_regular(wrapper)) != physical["implementation_sha256"][str(wrapper.relative_to(workspace))]:
        raise ValueError("Executed native command owner differs from the trusted source")
    add(wrapper, "native-command-owner")
    checked = validate_reference_placement_evidence(run, task_id,
        runtime_overlay=overlay if task_id == DYNAMIC_TASK else None)
    if task_id == DYNAMIC_TASK and not checked["dynamic_controls"]["identity_controls_only"]:
        raise ValueError("Dynamic reference requires the identity initial controller")
    _placement_def(run / STAGE / "placement.gp.def", run / "20_partition/partition_assignment.csv")
    raw = _strict_native_metrics(run)
    required_metrics = {"placement_native_hpwl", "placement_overflow_bottom", "placement_overflow_top",
                        "placement_max_density_bottom", "placement_max_density_top"}
    if not required_metrics.issubset(raw) or any(isinstance(value, bool) or not math.isfinite(value) or value < 0
                                               for value in raw.values()):
        raise ValueError("Native placement lacks complete finite two-die measurements")
    return {"qualified": True, "producer_profile": "native-local-placement-v1", "case": case,
            "endpoint": ENDPOINTS[task_id], "baseline": {"raw_metrics": raw, "source_run": str(run),
                "native_runtime_historical": {"seconds": seconds, "scope": "native placement flow process"}},
            "source_identity": {"purpose_sha256": file_sha(receipt_path), "image": physical["image"],
                "source_revision": physical["source_revision"], "runtime_source_audit_sha256": file_sha(audit_path)},
            "source_contract": physical, "frozen_inputs": frozen,
            "evidence": [evidence[key] for key in sorted(evidence)], "executed_native_stages": []}
