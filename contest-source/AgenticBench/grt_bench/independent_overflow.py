"""Opt-in independent audit of the self-reported post-GRT overflow.

The ``overflow`` objective is FastRoute's own final table (``grt.log``).  When
FastRoute is candidate-editable that number is self-reported.  This module
recomputes an independent estimate from the guide and DEF written by the
frozen flow (``grt_bench.guide_congestion``, part of the trusted evaluator
mounted at ``/trusted``, never the candidate workspace) and compares the two
under one declared, explicit rule.  Frozen contracts are unaffected: the
verifier calls :func:`verify_hook` only when a task carries an
``independent_overflow`` block.

Consistency rule (version 1), with R = reported and E = estimated overflow:

* bound: per die and in total, ``E <= R * (1 + rel_tolerance) + abs_tolerance``.
  The estimate under-approximates FastRoute's overflow (more generous
  capacity, at most one unit of demand per net and edge), so a reported value
  far below it means the reported overflow is not supported by the guides.
* optional ``require_positive_estimate``: ``R > 0`` requires ``E > 0``.
* reference (when a reference ``{reported, estimate}`` is given): a claimed
  improvement ``R < R_ref`` requires
  ``E <= E_ref * (1 + reference_rel_tolerance) + reference_abs_tolerance``;
  the independent estimate must not contradict the claimed direction.

Every missing, symlinked, unparsable or hash-mismatched artifact fails closed
(``consistent: false`` with an error), never silently passes.
"""
import math
import re
from pathlib import Path

from .common import read, sha

AUDIT_NAME = "grt-independent-overflow"
AUDIT_VERSION = 1
DIES = ("bottom", "top")
AUDITED_ARTIFACTS = ("grt.log", "grt.tcl", "grt.guide", "global_route.odb.def")
DEFAULT_RULE = {
    "rel_tolerance": 0.05,
    "abs_tolerance": 5,
    "require_positive_estimate": False,
    "reference_rel_tolerance": 0.05,
    "reference_abs_tolerance": 5,
}
CONTRACT_KEYS = {"version", "enforce", "estimator_sha256", "track_model", "rule", "reference_estimates"}
_LAYER_ROW = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+([0-9.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$", re.M)


def _estimator():
    from . import guide_congestion
    return guide_congestion


def estimator_identity():
    module = _estimator()
    return {"name": module.ESTIMATOR_NAME, "version": module.ESTIMATOR_VERSION,
            "sha256": module.source_sha256(), "audit_version": AUDIT_VERSION, "audit_sha256": sha(__file__)}


def validate_rule(rule=None):
    """Return the complete rule; reject unknown keys and invalid values."""
    if rule is not None and not isinstance(rule, dict):
        raise ValueError("Independent overflow rule must be an object")
    rule = dict(DEFAULT_RULE, **(rule or {}))
    unknown = set(rule) - set(DEFAULT_RULE)
    if unknown:
        raise ValueError("Unknown independent overflow rule keys: " + ", ".join(sorted(unknown)))
    if not isinstance(rule["require_positive_estimate"], bool):
        raise ValueError("require_positive_estimate must be a boolean")
    for key in sorted(set(rule) - {"require_positive_estimate"}):
        value = rule[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid independent overflow tolerance: " + key)
    return rule


def _count(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Invalid overflow count: " + label)
    return value


def _reference(value, label):
    if not isinstance(value, dict) or set(value) != {"reported", "estimate"}:
        raise ValueError("Reference must be {reported, estimate}: " + label)
    return {key: _count(value[key], label + "." + key) for key in ("reported", "estimate")}


def validate_contract(block):
    """Validate a task's ``independent_overflow`` block and fill defaults.

    ``{"version": 1, "enforce": true, "estimator_sha256": "<hex>",
    "track_model": "average", "rule": {...},
    "reference_estimates": {"<case>": {"<seed>": {"reported": R, "estimate": E}}}}``
    """
    if not isinstance(block, dict) or block.get("version") != 1:
        raise ValueError("independent_overflow requires version 1")
    unknown = set(block) - CONTRACT_KEYS
    if unknown:
        raise ValueError("Unknown independent_overflow keys: " + ", ".join(sorted(unknown)))
    enforce = block.get("enforce", True)
    if not isinstance(enforce, bool):
        raise ValueError("independent_overflow.enforce must be a boolean")
    pinned = block.get("estimator_sha256")
    if pinned is not None and not re.fullmatch(r"[0-9a-f]{64}", str(pinned)):
        raise ValueError("independent_overflow.estimator_sha256 must be a SHA-256")
    track_model = block.get("track_model", "average")
    if track_model not in _estimator().TRACK_MODELS:
        raise ValueError("Unknown independent_overflow.track_model")
    references = block.get("reference_estimates", {})
    if not isinstance(references, dict):
        raise ValueError("independent_overflow.reference_estimates must be an object")
    normalized = {}
    for case, seeds in references.items():
        if not isinstance(seeds, dict):
            raise ValueError("Reference estimates must map case -> seed -> reference")
        normalized[str(case)] = {str(seed): _reference(value, "{}/{}".format(case, seed))
                                 for seed, value in seeds.items()}
    return {"version": 1, "enforce": enforce, "estimator_sha256": pinned, "track_model": track_model,
            "rule": validate_rule(block.get("rule")), "reference_estimates": normalized}


def reported_overflow(die_dir):
    """Return the scored overflow (read exactly as three_d_route does) and the per-layer table."""
    from open3dbench.backend.three_d_route import _grt_overflow_count
    log = Path(die_dir) / "grt.log"
    total = _grt_overflow_count(log)
    text = log.read_text(encoding="utf-8", errors="replace")
    marker = text.rfind("Final congestion report")
    layers = {}
    for row in _LAYER_ROW.finditer(text[marker:] if marker >= 0 else text):
        if row.group(1) != "Total":
            layers[row.group(1)] = {"resource": int(row.group(2)), "demand": int(row.group(3)),
                                    "overflow": int(row.group(7))}
    return total, layers


def _check_artifacts(die_dir, require_manifest):
    for name in AUDITED_ARTIFACTS:
        path = die_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked GRT artifact: " + str(path))
    if not require_manifest:
        return None
    manifest_path = die_dir / "grt_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Missing or symlinked GRT manifest: " + str(manifest_path))
    manifest = read(manifest_path)
    if manifest.get("stage") != "global_route" or manifest.get("status") != "success":
        raise ValueError("GRT manifest is not a successful global_route seal: " + str(manifest_path))
    for name in AUDITED_ARTIFACTS:
        record = manifest.get("artifacts", {}).get(name)
        if not isinstance(record, dict) or record.get("sha256") != sha(die_dir / name):
            raise ValueError("GRT artifact differs from its sealed manifest: " + str(die_dir / name))
    return manifest


def audit_die(die_dir, *, platform=None, tech_lef=None, layer_directions=None, gcell_size=None,
              track_model="average", require_manifest=True):
    """Audit one die directory; raises ``ValueError`` on any evidence problem."""
    die_dir = Path(die_dir)
    manifest = _check_artifacts(die_dir, require_manifest)
    reported, reported_layers = reported_overflow(die_dir)
    if manifest is not None and manifest.get("metrics", {}).get("grt_total_overflow") != reported:
        raise ValueError("Sealed GRT metrics differ from the log overflow: " + str(die_dir))
    estimate = _estimator().estimate_die(
        die_dir, tech_lef=tech_lef, platform=platform, layer_directions=layer_directions,
        gcell_size=gcell_size, track_model=track_model)
    summary = {key: estimate[key] for key in (
        "total_overflow", "max_edge_overflow", "overflowing_edges", "capacity", "demand", "nets", "rects",
        "single_gcell_rects", "nonpreferred_rects", "rects_outside_die", "duplicate_net_blocks",
        "direction_source", "layer_range", "inputs", "tech_lef", "grid")}
    summary["layers"] = {name: {key: layer[key] for key in ("direction", "capacity", "demand", "overflow",
                                                              "max_edge_overflow", "overflowing_edges")}
                         for name, layer in estimate["layers"].items()}
    return {"reported_overflow": reported, "reported_layers": reported_layers,
            "estimated_overflow": estimate["total_overflow"], "estimate": summary,
            "manifest_verified": manifest is not None}


def _bound(scope, reported, estimate, rule):
    limit = reported * (1 + rule["rel_tolerance"]) + rule["abs_tolerance"]
    return {"check": "bound", "scope": scope, "reported": reported, "estimate": estimate,
            "limit": limit, "passed": estimate <= limit}


def evaluate(dies, rule, reference=None):
    """Apply the declared rule to ``{die: (reported, estimate)}``; return the checks."""
    rule = validate_rule(rule)
    checks = []
    for die, (reported, estimate) in sorted(dies.items()):
        checks.append(_bound(die, reported, estimate, rule))
        if rule["require_positive_estimate"]:
            checks.append({"check": "positive", "scope": die, "reported": reported, "estimate": estimate,
                           "passed": reported == 0 or estimate > 0})
    reported = sum(value[0] for value in dies.values())
    estimate = sum(value[1] for value in dies.values())
    if len(dies) > 1:
        checks.append(_bound("total", reported, estimate, rule))
    if reference is not None:
        claimed = reported < reference["reported"]
        limit = reference["estimate"] * (1 + rule["reference_rel_tolerance"]) + rule["reference_abs_tolerance"]
        checks.append({"check": "reference", "scope": "total", "reported": reported, "estimate": estimate,
                       "reference": reference, "claimed_improvement": claimed, "limit": limit,
                       "passed": (not claimed) or estimate <= limit})
    return checks


def load_reference(reference):
    """Accept ``{reported, estimate}`` or the path of a consistent receipt."""
    if reference is None:
        return None
    if isinstance(reference, (str, Path)):
        data = read(reference)
        if data.get("audit") != AUDIT_NAME:
            raise ValueError("Reference is not an independent overflow receipt: " + str(reference))
        if data.get("consistent") is not True:
            raise ValueError("Reference receipt is not itself consistent: " + str(reference))
        reference = {"reported": data["reported_overflow"], "estimate": data["estimated_overflow"]}
    return _reference(reference, "reference")


def _die_dirs(grt_dir):
    grt_dir = Path(grt_dir)
    if not any((grt_dir / die).exists() for die in DIES) and (grt_dir / "grt.guide").exists():
        return [(grt_dir.name, grt_dir)]
    return [(die, grt_dir / die) for die in DIES]


def _error(scope, error):
    return {"scope": scope, "error": "{}: {}".format(type(error).__name__, error)}


def audit_grt(grt_dir, *, rule=None, reference=None, platform=None, tech_lef=None, layer_directions=None,
              gcell_size=None, track_model="average", require_manifest=True):
    """Audit a GRT output directory (``bottom/`` + ``top/``, or one die directory).

    Returns a JSON-serializable receipt.  Errors are recorded rather than
    raised and always make the receipt inconsistent (fail closed).
    """
    receipt = {"schema_version": 1, "audit": AUDIT_NAME, "grt_dir": str(grt_dir), "dies": {}, "errors": [],
               "checks": [], "consistent": False, "reported_overflow": None, "estimated_overflow": None}
    try:
        receipt["estimator"] = estimator_identity()
        rule = validate_rule(rule)
        receipt["rule"] = rule
        receipt["reference"] = reference = load_reference(reference)
    except Exception as error:  # noqa: BLE001 - recorded, fail closed
        receipt["errors"].append(_error("setup", error))
        return receipt
    values = {}
    for die, directory in _die_dirs(grt_dir):
        try:
            result = audit_die(directory, platform=platform, tech_lef=tech_lef, layer_directions=layer_directions,
                               gcell_size=gcell_size, track_model=track_model, require_manifest=require_manifest)
        except Exception as error:  # noqa: BLE001 - recorded, fail closed
            receipt["errors"].append(_error(die, error))
            continue
        receipt["dies"][die] = result
        values[die] = (result["reported_overflow"], result["estimated_overflow"])
    if receipt["errors"]:
        return receipt
    receipt["reported_overflow"] = sum(value[0] for value in values.values())
    receipt["estimated_overflow"] = sum(value[1] for value in values.values())
    receipt["checks"] = evaluate(values, rule, reference)
    receipt["consistent"] = all(check["passed"] for check in receipt["checks"])
    return receipt


def _seed_id(seed):
    return str(seed.get("id")) if isinstance(seed, dict) else str(seed)


def verify_hook(task, plan, output):
    """Verifier hook for tasks that declare ``independent_overflow``.

    Returns the receipt (stored under the verified receipt's
    ``independent_overflow`` key) and raises ``ValueError`` when the block is
    enforced and the audit is inconsistent or failed closed.
    """
    contract = validate_contract(task["independent_overflow"])
    identity = estimator_identity()
    if contract["estimator_sha256"] is not None and contract["estimator_sha256"] != identity["sha256"]:
        raise ValueError("Independent overflow estimator differs from the task")
    manifest = read(Path(output) / "analysis/manifest.json")
    grt_dir = Path(manifest["source_checkpoint"]).parent
    reference = contract["reference_estimates"].get(str(plan["case"]), {}).get(_seed_id(plan["seed"]))
    receipt = audit_grt(grt_dir, rule=contract["rule"], reference=reference, platform=manifest.get("platform"),
                        track_model=contract["track_model"])
    receipt["enforced"] = contract["enforce"]
    if contract["enforce"] and not receipt["consistent"]:
        reasons = [check["scope"] + ":" + check["check"] for check in receipt["checks"] if not check["passed"]]
        reasons += [error["scope"] + ": " + error["error"] for error in receipt["errors"]]
        raise ValueError("Independent overflow audit failed: " + "; ".join(reasons))
    return receipt


def _is_grt(path):
    return any((path / die / "grt.guide").is_file() or (path / die / "grt.log").is_file() for die in DIES)


def discover(root):
    """Return GRT directories: ``root`` itself, ``root/grt``, or every nested one."""
    root = Path(root)
    if _is_grt(root) or (root / "grt.guide").is_file():
        return [root]
    if _is_grt(root / "grt"):
        return [root / "grt"]
    found = set()
    for name in ("grt.guide", "grt.log"):
        for path in root.rglob(name):
            if path.parent.name in DIES and not path.is_symlink():
                found.add(path.parent.parent)
    return sorted(found)
