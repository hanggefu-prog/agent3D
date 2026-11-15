#!/usr/bin/env python3
"""Check that the maintained task documentation matches the code.

The code is the source of truth. This script derives these facts from the
checkout it lives in:

- the maintained task inventory (``task_inventory.CORE_TASK_DIRECTORIES``
  loaded through ``model.load_task``);
- the score weights in ``local_bench.metrics`` and the behaviour of
  ``hbt_policy.score_timing``;
- the post-GRT timing weights and power limit in
  ``grt_bench.contract.DEFAULT_SCORING``, read only;
- the default matrix profile and CPU smoke input IDs.

It then checks the root README task table and the owned AgenticBench docs
against those facts:

- TASK_SETTINGS_REPORT.md
- HARNESS_DESIGN_REPORT.md
- LOCAL_EVALUATION_MATRICES.md
- tasks.md

It also checks that relative links in the owned docs resolve.

Usage::

    python AgenticBench/scripts/check_doc_consistency.py [--repo-root PATH]

The script exits 1 and prints one line per mismatch when anything differs.
Tests call ``check(repo_root, facts=..., check_links=...)`` directly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (SCRIPT_REPO_ROOT / "AgenticBench", SCRIPT_REPO_ROOT, SCRIPT_REPO_ROOT / "Place-3D"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

DOCS = "AgenticBench/docs"
TASK_REPORT = DOCS + "/TASK_SETTINGS_REPORT.md"
HARNESS_REPORT = DOCS + "/HARNESS_DESIGN_REPORT.md"
MATRIX_DOC = DOCS + "/LOCAL_EVALUATION_MATRICES.md"
TASKS_DOC = DOCS + "/tasks.md"
ROOT_README = "README.md"
OWNED_DOCS = (TASK_REPORT, HARNESS_REPORT, MATRIX_DOC, TASKS_DOC)
ROUTING_WRAPPERS = ("open3dbench/backend/three_d_route.py", "open3dbench/runner.py")
SMOKE_INPUT = "ibex_lol_ng45_cpu_smoke"


def _num(value: float) -> str:
    """Format a weight the way the docs write it (0.6, 0.9, 1.1)."""
    return "{:g}".format(float(value))


# ---------------------------------------------------------------------------
# Facts from code
# ---------------------------------------------------------------------------

def collect_code_facts() -> Dict[str, Any]:
    """Return the facts the docs must agree with, derived from code only."""

    from local_bench import metrics
    from local_bench.hbt_policy import score_timing
    from local_bench.matrix_profile import load_matrix_profile
    from local_bench.benchmark_input import load_benchmark_input_profile
    from local_bench.model import load_task
    from local_bench.task_inventory import CORE_TASK_DIRECTORIES

    task_root = SCRIPT_REPO_ROOT / "AgenticBench" / "local_tasks"
    tasks = []
    for directory in CORE_TASK_DIRECTORIES:
        spec = load_task(task_root / directory)
        tasks.append({
            "directory": directory,
            "id": spec.task_id,
            "kind": spec.kind,
            "entrypoint": spec.program_entrypoint,
            "start": spec.start_stage,
            "stop": spec.stop_stage,
            "upstream": list(spec.upstream_stages),
            "source_paths": list(spec.allowed_source_paths),
            "config_paths": list(spec.allowed_config_paths),
            "candidate_files": [item.path for item in spec.candidate_files],
            "limits": (spec.max_changed_files, spec.max_added_lines, spec.max_deleted_lines),
            "scoring": dict(spec.scoring),
        })

    # Derive the HBT weights from behaviour: a unit WNS cost and a zero TNS
    # cost leave exactly the WNS weight.
    hbt = score_timing({"wns_setup": -1.0, "tns_setup": 0.0},
                       {"wns_setup": -1.0, "tns_setup": -1.0})
    hbt_wns_weight = hbt["normalized_sta_cost"]

    try:
        from grt_bench.contract import DEFAULT_SCORING
        post_grt: Optional[Dict[str, float]] = {
            "wns": float(DEFAULT_SCORING["timing_weights"]["wns_deficit_ns"]),
            "tns": float(DEFAULT_SCORING["timing_weights"]["tns_deficit_ns"]),
            "power_max_ratio": float(DEFAULT_SCORING["power_max_ratio"]),
        }
        post_grt_error = None
    except Exception as error:  # pragma: no cover - reported as a finding
        post_grt, post_grt_error = None, "{}: {}".format(type(error).__name__, error)

    profile = load_matrix_profile()
    smoke = load_benchmark_input_profile(SMOKE_INPUT)
    return {
        "tasks": tasks,
        "sta_wns_weight": float(metrics.STA_WNS_WEIGHT),
        "sta_tns_weight": float(metrics.STA_TNS_WEIGHT),
        "hbt_wns_weight": float(hbt_wns_weight),
        "placement_hpwl_weight": float(metrics.PLACEMENT_HPWL_WEIGHT),
        "placement_overflow_weight": float(metrics.PLACEMENT_OVERFLOW_WEIGHT),
        "routing_overflow_weight": float(metrics.ROUTING_ORDER_OVERFLOW_WEIGHT),
        "routing_wirelength_weight": float(metrics.ROUTING_ORDER_WIRELENGTH_WEIGHT),
        "post_grt": post_grt,
        "post_grt_error": post_grt_error,
        "matrix_profile_id": profile["id"],
        "matrix_profile_qualified": profile["qualified"],
        "smoke_input_id": smoke.profile_id,
        "smoke_compatible_tasks": list(smoke.compatible_tasks),
    }


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _cells(line: str) -> List[str]:
    return [cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")]


def table_rows(text: str, header_first_cell: str) -> List[List[str]]:
    """Return body rows of the first table whose header starts with a cell."""

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith("|") and _cells(line)[0] == header_first_cell:
            rows = []
            for body in lines[index + 2:]:
                if not body.lstrip().startswith("|"):
                    break
                rows.append(_cells(body))
            return rows
    return []


def markdown_section(text: str, heading_prefix: str) -> Optional[str]:
    """Text under the first heading whose title starts with ``heading_prefix``.

    The section ends at the next heading of the same or a higher level.
    """

    lines = text.splitlines()
    fenced = []  # True for lines inside ``` code blocks, whose '#' is not a heading
    inside = False
    for line in lines:
        if line.lstrip().startswith("```"):
            inside = not inside
            fenced.append(True)
        else:
            fenced.append(inside)
    for index, line in enumerate(lines):
        match = None if fenced[index] else re.match(r"^(#+)\s+(.*)$", line)
        if match and match.group(2).startswith(heading_prefix):
            level = len(match.group(1))
            body = []
            for offset, following in enumerate(lines[index + 1:], index + 1):
                other = None if fenced[offset] else re.match(r"^(#+)\s", following)
                if other and len(other.group(1)) <= level:
                    break
                body.append(following)
            return "\n".join(body)
    return None


_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def broken_links(path: Path) -> List[str]:
    """Relative markdown links in ``path`` whose target file does not exist."""

    errors = []
    in_code = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for target in _LINK.findall(line):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            file_part = target.split("#", 1)[0]
            if file_part and not (path.parent / file_part).exists():
                errors.append("{}:{}: broken link {}".format(path.name, number, target))
    return errors


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _require_text(errors: List[str], name: str, text: str, needle: str, why: str) -> None:
    # Compare with whitespace normalized so that line wrapping never matters.
    if " ".join(needle.split()) not in " ".join(text.split()):
        errors.append("{}: missing {!r} ({})".format(name, needle, why))


def _stage_cell_errors(name: str, task: Dict[str, Any], cell: str) -> List[str]:
    errors = []
    match = re.match(r"^(\S+)(?: through (\S+))? \((.*)\)$", cell)
    if not match:
        return ["{}: stage cell for {} is not 'start [through stop] (upstream)': {!r}".format(
            name, task["id"], cell)]
    start, stop, upstream = match.group(1), match.group(2) or match.group(1), match.group(3)
    if (start, stop) != (task["start"], task["stop"]):
        errors.append("{}: {} runs {} through {} in code, doc says {} through {}".format(
            name, task["id"], task["start"], task["stop"], start, stop))
    stages = task["upstream"]
    accepted = {", ".join(stages)}
    if stages and len(stages) > 3:
        accepted.add("{} through {}".format(stages[0], stages[-1]))
    if upstream not in accepted:
        errors.append("{}: {} upstream stages are {} in code, doc says ({})".format(
            name, task["id"], ", ".join(stages), upstream))
    return errors


def check_task_report(text: str, facts: Dict[str, Any]) -> List[str]:
    name = TASK_REPORT
    errors: List[str] = []
    tasks = facts["tasks"]

    rows = table_rows(text, "Directory")
    by_dir = {row[0]: row for row in rows if row}
    if set(by_dir) != {task["directory"] for task in tasks} or len(rows) != len(tasks):
        errors.append("{}: inventory table lists {} but code has {}".format(
            name, sorted(by_dir), sorted(task["directory"] for task in tasks)))
    for task in tasks:
        row = by_dir.get(task["directory"])
        if row is None or len(row) < 5:
            continue
        for label, column, value in (("task ID", 1, task["id"]), ("kind", 2, task["kind"]),
                                     ("entrypoint", 3, task["entrypoint"])):
            if row[column] != value:
                errors.append("{}: {} {} is {!r} in code, doc says {!r}".format(
                    name, task["directory"], label, value, row[column]))
        errors.extend(_stage_cell_errors(name, task, row[4]))

    scope_rows = table_rows(text, "Task ID")
    by_id = {row[0]: row for row in scope_rows if row}
    if set(by_id) != {task["id"] for task in tasks}:
        errors.append("{}: edit-scope table lists {} but code has {}".format(
            name, sorted(by_id), sorted(task["id"] for task in tasks)))
    previous_config: Optional[Tuple[str, ...]] = None
    for task in tasks:
        row = by_id.get(task["id"])
        config_names = tuple(task["config_paths"] + task["candidate_files"])
        if row is None or len(row) < 4:
            previous_config = config_names
            continue
        limits = " / ".join(str(value) for value in task["limits"])
        if row[3] != limits:
            errors.append("{}: {} limits (files / added / deleted) are {} in code, doc says {}".format(
                name, task["id"], limits, row[3]))
        sources = [path for path in task["source_paths"] if path not in ROUTING_WRAPPERS]
        if not task["source_paths"]:
            if row[1] != "none":
                errors.append("{}: {} has no editable source paths, doc says {!r}".format(
                    name, task["id"], row[1]))
        for path in sources:
            if Path(path).name not in row[1]:
                errors.append("{}: {} editable source {} missing from the edit-scope table".format(
                    name, task["id"], path))
        if any(path in ROUTING_WRAPPERS for path in task["source_paths"]) and "wrappers" not in row[1]:
            errors.append("{}: {} lists the evaluator wrappers in code; the doc does not".format(
                name, task["id"]))
        if row[2] == "same as above":
            if config_names != previous_config:
                errors.append("{}: {} config/candidate files differ from the previous row".format(
                    name, task["id"]))
        elif not config_names:
            if row[2] != "none":
                errors.append("{}: {} has no config or candidate files, doc says {!r}".format(
                    name, task["id"], row[2]))
        else:
            for item in config_names:
                if item not in row[2]:
                    errors.append("{}: {} config/candidate file {} missing from the doc".format(
                        name, task["id"], item))
        previous_config = config_names

    wns, tns = _num(facts["sta_wns_weight"]), _num(facts["sta_tns_weight"])
    hpwl, ovf = _num(facts["placement_hpwl_weight"]), _num(facts["placement_overflow_weight"])
    rovf, rwl = _num(facts["routing_overflow_weight"]), _num(facts["routing_wirelength_weight"])
    placement = "{} * hpwl_score + {} * overflow_score".format(hpwl, ovf)
    for heading, needle, why in (
        ("TP-GNN post-placement timing", "{} * cost_W + {} * cost_T".format(wns, tns),
         "TP-GNN timing formula, metrics.STA_*"),
        ("TP-GNN post-placement timing", "sta_wns_weight = {}".format(wns), "reported TP-GNN weight key"),
        ("Placement initialization", placement, "placement formula, metrics.PLACEMENT_*"),
        ("Dynamic Nesterov control", placement, "placement formula, metrics.PLACEMENT_*"),
        ("HBT assignment policy", "{} * rel(dW, dW_ref) + {} * rel(dT, dT_ref)".format(wns, tns),
         "HBT timing formula, hbt_policy.score_timing"),
        ("Global-routing source tasks", "{} * overflow_cost + {} * wirelength_cost".format(rovf, rwl),
         "routing v1 formula, metrics.ROUTING_ORDER_*"),
        ("Two timing-weight conventions", "{} WNS / {} TNS".format(wns, tns), "local timing weights"),
    ):
        section = markdown_section(text, heading)
        if section is None:
            errors.append("{}: missing section starting with {!r}".format(name, heading))
        else:
            _require_text(errors, name + " section " + repr(heading), section, needle, why)
    for needle, why in (
        ("default_matrix_profile.yaml", "matrix profile file"),
        (facts["smoke_input_id"], "CPU smoke input id"),
    ):
        _require_text(errors, name, text, needle, why)
    errors.extend(_post_grt_errors(name, text, facts, "{wns} WNS / {tns} TNS"))
    return errors


def _post_grt_errors(name: str, text: str, facts: Dict[str, Any], pattern: str) -> List[str]:
    post = facts.get("post_grt")
    if post is None:
        return ["{}: cannot read grt_bench.contract.DEFAULT_SCORING ({})".format(
            name, facts.get("post_grt_error"))]
    errors: List[str] = []
    _require_text(errors, name, text, pattern.format(wns=_num(post["wns"]), tns=_num(post["tns"])),
                  "post-GRT timing weights, grt_bench.contract.DEFAULT_SCORING")
    return errors


def check_harness_report(text: str, facts: Dict[str, Any]) -> List[str]:
    name = HARNESS_REPORT
    flat = " ".join(text.split())  # sentences may wrap anywhere
    errors = _post_grt_errors(name, flat, facts, "WNS {wns} and TNS {tns}")
    _require_text(errors, name, flat, "weight them {} and {}".format(
        _num(facts["sta_wns_weight"]), _num(facts["sta_tns_weight"])), "local timing weights")
    for function in ("placement_component_metrics", "placement_nesterov_metrics",
                     "_routing_source_policy_metrics"):
        _require_text(errors, name, text, function, "metric extractor named in metrics.py")
    return errors


def check_matrix_doc(text: str, facts: Dict[str, Any]) -> List[str]:
    name = MATRIX_DOC
    errors: List[str] = []
    _require_text(errors, name, text, facts["matrix_profile_id"], "default matrix profile id")
    _require_text(errors, name, text, "`{}` overflow and `{}`".format(
        _num(facts["routing_overflow_weight"]), _num(facts["routing_wirelength_weight"])),
        "routing v1 weights")
    if facts["matrix_profile_qualified"] is not False:
        errors.append("{}: default matrix profile must declare qualified: false".format(name))
    return errors


def check_tasks_doc(text: str, facts: Dict[str, Any]) -> List[str]:
    name = TASKS_DOC
    errors: List[str] = []
    rows = table_rows(text, "Task directory")
    listed = [row[0] for row in rows if row]
    expected = [task["directory"] for task in facts["tasks"]]
    if sorted(listed) != sorted(expected):
        errors.append("{}: task table lists {} but code has {}".format(name, sorted(listed), sorted(expected)))
    _require_text(errors, name, text, "{} WNS / {} TNS".format(
        _num(facts["sta_wns_weight"]), _num(facts["sta_tns_weight"])), "local timing weights")
    errors.extend(_post_grt_errors(name, text, facts, "{wns} WNS / {tns} TNS"))
    _require_text(errors, name, text, "## 5-minute path for external users", "external smoke section")
    _require_text(errors, name, text, facts["smoke_input_id"], "CPU smoke input id")
    for task_id in facts["smoke_compatible_tasks"]:
        _require_text(errors, name, text, "evaluate " + task_id, "smoke-compatible task command")
    return errors


def check_root_readme(text: str, facts: Dict[str, Any]) -> List[str]:
    name = ROOT_README
    errors: List[str] = []
    rows = table_rows(text, "Track")
    local = [row[1] for row in rows if row and row[0].startswith("Local")]
    expected = [task["id"] for task in facts["tasks"]]
    if sorted(local) != sorted(expected):
        errors.append("{}: local task table lists {} but code task IDs are {}".format(
            name, sorted(local), sorted(expected)))
    flat = " ".join(text.split())
    _require_text(errors, name, flat, "{} WNS / {} TNS".format(
        _num(facts["sta_wns_weight"]), _num(facts["sta_tns_weight"])), "local HBT timing weights")
    post = facts.get("post_grt")
    if post is None:
        errors.extend(_post_grt_errors(name, flat, facts, ""))
    else:
        _require_text(errors, name, flat, "{} normalized WNS + {} normalized TNS".format(
            _num(post["wns"]), _num(post["tns"])), "post-GRT timing weights")
        _require_text(errors, name, flat, "{:g}%".format(round(post["power_max_ratio"] * 100, 6)),
                      "post-GRT power_max_ratio")
    return errors


def check_code_self_consistency(facts: Dict[str, Any]) -> List[str]:
    errors = []
    if abs(facts["hbt_wns_weight"] - facts["sta_wns_weight"]) > 1e-12:
        errors.append("code: hbt_policy.score_timing uses WNS weight {} but metrics.STA_WNS_WEIGHT is {}".format(
            facts["hbt_wns_weight"], facts["sta_wns_weight"]))
    if abs(facts["sta_wns_weight"] + facts["sta_tns_weight"] - 1.0) > 1e-12:
        errors.append("code: metrics.STA_WNS_WEIGHT + STA_TNS_WEIGHT != 1")
    return errors


CHECKS = (
    (TASK_REPORT, check_task_report),
    (HARNESS_REPORT, check_harness_report),
    (MATRIX_DOC, check_matrix_doc),
    (TASKS_DOC, check_tasks_doc),
    (ROOT_README, check_root_readme),
)


def check(repo_root: Path, facts: Optional[Dict[str, Any]] = None,
          check_links: bool = True) -> List[str]:
    """Return every documentation mismatch under ``repo_root``.

    ``facts`` defaults to :func:`collect_code_facts` from this checkout, so a
    test can check a modified copy of the docs against the real code.
    """

    repo_root = Path(repo_root)
    facts = collect_code_facts() if facts is None else facts
    errors = check_code_self_consistency(facts)
    for relative, function in CHECKS:
        path = repo_root / relative
        if not path.is_file():
            errors.append("{}: file is missing".format(relative))
            continue
        errors.extend(function(path.read_text(encoding="utf-8"), facts))
    if check_links:
        for relative in OWNED_DOCS:
            path = repo_root / relative
            if path.is_file():
                errors.extend(DOCS + "/" + item for item in broken_links(path))
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_REPO_ROOT)
    parser.add_argument("--no-links", action="store_true", help="skip relative link checks")
    args = parser.parse_args(argv)
    errors = check(args.repo_root, check_links=not args.no_links)
    for error in errors:
        print(error)
    if errors:
        print("doc consistency: {} mismatch(es)".format(len(errors)), file=sys.stderr)
        return 1
    print("doc consistency: OK ({} files checked)".format(len(CHECKS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
