"""Versioned evaluation profiles for the engineering track.

scoring_version 1 is the frozen contract of existing qualifications/receipts.
scoring_version 2 adds the gates and calibration declared in
engineering_tasks/<task>/evaluation-v2.json. The profile lives outside
task.json on purpose: task.json (and *.md) are part of task.digest(), which old
receipts bind, while a v2 qualification binds the profile by its own sha256.
Expert QoR is report-time calibration and lives in a separate registry, so
registering an expert never invalidates a physical receipt.
"""
import copy
import json
import math
import re

from .protocol import sha
from .tasks import read_json, relative

SCORING_VERSIONS = (1, 2)
CURRENT_SCORING_VERSION = 1
PROFILE_NAME = "evaluation-v2.json"

# Keep the legacy prompt and CTS evaluator on the same frozen policy. Version 2
# is an explicit new contract, not an implicit upgrade of existing task receipts.
LEGACY_CTS_POLICY = {
    'id': 'engineering-cts-final-guards-v1',
    'clock_area_max_reference_ratio': 2.0,
    'clock_buffer_count_max_reference_ratio': 2.0,
    'electrical': 'clock-only violation count and worst normalized excess nonregression',
    'electrical_ratio_tolerance': 1e-6,
    'setup_hold': 'diagnostic only',
    'native_runtime': 'source-bound installed candidate image; fixed-base independent verification',
    'physical_timeout_seconds': 3600,
}


def selected_contract(task, version=CURRENT_SCORING_VERSION):
    """Bind the prompt's actual scoring rules without rewriting frozen task files."""
    if type(version) is not int or version not in SCORING_VERSIONS:
        raise ValueError("Unknown engineering scoring version")
    result = {"schema": "engineering-selected-scoring-v1", "task": task.task_id,
              "task_digest": task.digest(), "scoring_version": version}
    if version == 2:
        profile, profile_sha = load_profile(task)
        result.update(scoring_profile_sha256=profile_sha, gates=profile['gates'])
    elif task.task_id == 'cross_die_cts':
        result['gates'] = copy.deepcopy(LEGACY_CTS_POLICY)
    else:
        result['gates'] = {'setup_hold_tolerance_ns': 1e-6,
            'new_hold_violating_endpoints_allowed': False,
            'resource_limits': 'the supplied case context',
            'electrical': 'violation count and worst normalized excess nonregression'}
        if task.task_id == 'multi_hbt_fanout':
            result['gates']['receiver_scope'] = 'original_remote_receivers'
    return result


def qualified_contract(task, proofs, version=CURRENT_SCORING_VERSION):
    """Fail before a developer call if any case has a different scoring contract."""
    contract = selected_contract(task, version)
    if not proofs or any(proof['scoring_version'] != version for proof in proofs):
        raise ValueError('Qualifications have another or mixed scoring version')
    for proof in proofs:
        profile_for(task, proof['record'])
    return contract


def scoring_prompt(task, contract):
    """Render thresholds directly from the contract used by the final evaluator."""
    if contract != selected_contract(task, contract['scoring_version']):
        raise ValueError('Selected scoring contract changed')
    lines = ["# Selected scoring contract", "",
             "Scoring version: %d." % contract['scoring_version']]
    if contract['scoring_version'] == 2:
        lines.append('Evaluation profile SHA-256: ' + contract['scoring_profile_sha256'] + '.')
        if task.task_id == 'cross_die_cts':
            lines.append('Setup/hold WNS/TNS are acceptance gates under the tolerances below. '
                         'Insertion delay and clock resources must also satisfy these bounds. '
                         'Clock power is gated only when independently measured.')
        elif task.task_id == 'multi_hbt_fanout':
            lines.append('Measure all original receivers, including local and remote receivers. '
                         'The context must list original_local_receivers for every target. '
                         'The added-buffer area bound below also applies.')
    elif task.task_id == 'cross_die_cts':
        gates = contract['gates']
        lines.append('Setup/hold is diagnostic only. Clock area and buffer count must be '
                     'at most %gx and %gx reference, respectively.' % (
                         gates['clock_area_max_reference_ratio'], gates['clock_buffer_count_max_reference_ratio']))
    elif task.task_id == 'multi_hbt_fanout':
        lines.append('The objective measures original remote receivers; legacy contexts '
                     'without original_local_receivers remain valid.')
    lines += ['', 'The host checks the following rules and the supplied context limits:',
              json.dumps(contract['gates'], sort_keys=True, indent=2)]
    return '\n'.join(lines) + '\n'


def development_instruction(task, contract):
    """Avoid presenting the frozen v1 CTS timing sentence as a v2 instruction."""
    text = task.instruction
    if task.task_id == 'cross_die_cts' and contract['scoring_version'] == 2:
        legacy = ('Setup/hold WNS/TNS, when available from the same STA, are\n'
                  'diagnostics, not a second objective or an acceptance gate for this task.')
        if text.count(legacy) != 1:
            raise ValueError('Unknown legacy CTS instruction; review the v2 prompt')
        text = text.replace(legacy, 'Setup/hold WNS/TNS must satisfy the selected scoring '
                            'profile below. They are acceptance gates, not additional objectives.')
    return text


def validate_development_binding(task, subject, version):
    """A generated developer packet cannot silently change profile at evaluation."""
    from pathlib import Path
    from .protocol import digest
    folder = Path(subject).parent / 'control'
    path = folder / 'scoring-contract.json'
    if path.exists():
        contract = read_json(path)
        if contract != selected_contract(task, version):
            raise ValueError('Development and evaluation scoring contracts differ')
        environment = read_json(folder / 'environment.json')
        if (environment.get('scoring_contract_sha256') != sha(path)
                or environment.get('scoring_contract_digest') != digest(contract)
                or environment.get('prompt_sha256') != sha(folder / 'effective-prompt.txt')):
            raise ValueError('Development scoring/prompt binding changed')
    elif (folder / 'effective-prompt.txt').exists() and version != 1:
        raise ValueError('Legacy development prompt requires scoring version 1')
    # Hand-authored source has no development packet; its native qualification
    # still binds the explicit version and, for v2, the packaged profile hash.
    return selected_contract(task, version)


def scoring_version(record):
    """A missing field means a frozen version-1 record."""
    version = record.get("scoring_version", 1)
    if type(version) is not int or version not in SCORING_VERSIONS:
        raise ValueError("Unknown engineering scoring version")
    return version


def _ratio(value, label, minimum=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError("Invalid profile bound: " + label)
    return float(value)


def _nonnegative(value, label):
    return _ratio(value, label, 0.0)


def validate_profile(task, profile):
    required = {"schema", "task", "scoring_version", "identity_version", "capability_schema",
                "allow_unverified_capabilities", "calibration", "judge", "gates"}
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("Invalid engineering evaluation profile fields")
    if profile["schema"] != "engineering-evaluation-profile-v2" or profile["task"] != task.task_id:
        raise ValueError("Evaluation profile is for another task/schema")
    if profile["scoring_version"] != 2 or profile["identity_version"] != 2 or profile["capability_schema"] != 2:
        raise ValueError("Evaluation profile version mismatch")
    allowed = profile["allow_unverified_capabilities"]
    if not isinstance(allowed, list) or any(x not in task.data["required_capabilities"] for x in allowed):
        raise ValueError("Unverified allow-list must name required capabilities")
    calibration = profile["calibration"]
    if calibration.get("failed_case_score") != 0.0:
        raise ValueError("Failed cases must count as zero in the v2 headline")
    relative(calibration.get("expert_registry", ""))
    judge = profile["judge"]
    if judge.get("weight_in_headline") != 0:
        raise ValueError("The LLM judge must not contribute to the objective score")
    if type(judge.get("min_samples")) is not int or judge["min_samples"] < 3 or judge.get("aggregation") != "median":
        raise ValueError("The v2 judge needs at least three median-aggregated samples")
    gates = profile["gates"]
    if task.task_id == "cross_die_cts":
        tolerances = gates["timing"]["tolerances_ns"]
        if set(tolerances) != {"setup_wns_ns", "setup_tns_ns", "hold_wns_ns", "hold_tns_ns"}:
            raise ValueError("T4 timing tolerances must cover setup/hold WNS/TNS")
        for key, value in tolerances.items():
            _nonnegative(value, key)
        _nonnegative(gates["timing"]["relative_tolerance"], "relative_tolerance")
        _ratio(gates["insertion_delay"]["max_reference_ratio"], "insertion_delay")
        _nonnegative(gates["insertion_delay"]["absolute_tolerance_ns"], "insertion_delay tolerance")
        _ratio(gates["clock_area_max_reference_ratio"], "clock_area")
        _ratio(gates["clock_buffer_count_max_reference_ratio"], "clock_buffer_count")
        _ratio(gates["clock_power"]["max_reference_ratio"], "clock_power")
    elif task.task_id == "multi_hbt_fanout":
        if gates.get("receiver_scope") != "all":
            raise ValueError("T3 v2 measures every receiver of the target net")
        _ratio(gates["added_buffer_area"]["max_ratio_to_min_buffer_area_per_target"], "added_buffer_area", 0.0)
    return profile


def load_profile(task):
    """Return (profile, sha256) for the task's packaged v2 profile."""
    path = task.root / PROFILE_NAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing packaged v2 evaluation profile: " + str(path))
    return validate_profile(task, read_json(path)), sha(path)


def profile_for(task, record):
    """Profile a v2 record was qualified against; None for v1 records."""
    if scoring_version(record) == 1:
        return None
    profile, digest = load_profile(task)
    if record.get("scoring_profile_sha256") != digest:
        raise ValueError("Evaluation profile changed after qualification")
    return profile


def qualification_fields(task, version=CURRENT_SCORING_VERSION):
    """Fields a new qualification records about its scoring contract."""
    if version == 1:
        return {}
    if version != 2:
        raise ValueError("Unknown engineering scoring version")
    _, digest = load_profile(task)
    return {"scoring_version": 2, "scoring_profile_sha256": digest}


def expert_reference(task, profile):
    """Per-case expert QoR (None until registered) plus registry identity."""
    path = task.root / relative(profile["calibration"]["expert_registry"])
    registry = read_json(path)
    if registry.get("schema") != "engineering-expert-reference-v1" or registry.get("task") != task.task_id:
        raise ValueError("Invalid expert reference registry")
    cases = registry.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(task.data["cases"]):
        raise ValueError("Expert registry must list every case")
    result = {}
    for case, row in cases.items():
        if row is None:
            result[case] = None
            continue
        if not isinstance(row, dict) or set(row) != {"qor", "source", "receipt_sha256"}:
            raise ValueError("Invalid expert registration: " + case)
        if not isinstance(row["source"], str) or not row["source"].strip() or not re.fullmatch(r"[0-9a-f]{64}", str(row["receipt_sha256"])):
            raise ValueError("Expert registration needs a source and receipt hash: " + case)
        if isinstance(row["qor"], bool) or not isinstance(row["qor"], (int, float)) or not math.isfinite(row["qor"]) or row["qor"] < 0:
            raise ValueError("Invalid expert QoR: " + case)
        result[case] = float(row["qor"])
    return result, sha(path)
