"""Versioned score contracts for the local global-routing source tasks.

Version 1 is the historical default and is implemented directly in
``metrics._routing_source_policy_metrics``::

    overflow_cost   = (O + 1000) / (O_ref + 1000)
    wirelength_cost = W / W_ref
    combined_score  = grt_complete / max(0.1 * overflow_cost
                                         + 0.9 * wirelength_cost, 1e-12)

Version 2 is opt-in through ``task.yaml``::

    scoring:
      version: 2
      overflow_gate:
        max_ratio: 1.0      # optional, default 1.0
        abs_slack: 0        # required, total-overflow units

and replaces the smooth blend by an explicit overflow feasibility gate::

    gate passes  iff  O <= max_ratio * O_ref + abs_slack
    combined_score = grt_complete * W_ref / W   when the gate passes
    combined_score = 0                          when the gate fails

Existing task definitions declare no ``scoring`` block, so their task
contracts, prompts, scores and qualified references stay byte-identical.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple


ROUTING_SCORING_TASK_KINDS = frozenset(
    {
        "routing_ripup_policy_source",
        "routing_maze_cost_source",
        "routing_iteration_controller_source",
    }
)
SUPPORTED_ROUTING_SCORING_VERSIONS = (1, 2)
DEFAULT_OVERFLOW_GATE_MAX_RATIO = 1.0
# Numeric version-2 result keys exposed in optimization feedback.
ROUTING_V2_FEEDBACK_KEYS = frozenset(
    {
        "routing_scoring_version",
        "wirelength_cost",
        "wirelength_score",
        "overflow_gate_passed",
        "overflow_gate_evaluated",
        "overflow_gate_limit",
        "overflow_gate_max_ratio",
        "overflow_gate_abs_slack",
        "reference_grt_total_overflow",
        "reference_grt_wirelength_um",
    }
)


def _finite_non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a finite non-negative number".format(label))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be a finite non-negative number".format(label))
    return result


def parse_scoring(value: Any, kind: str, *, source: str = "task.yaml") -> Dict[str, Any]:
    """Validate a ``scoring`` block and return its normalized form.

    Absent blocks and explicit ``version: 1`` normalize to ``{}`` so the
    default contract representation is unchanged.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("{}: scoring must be an object".format(source))
    unknown = sorted(set(value) - {"version", "overflow_gate"})
    if unknown:
        raise ValueError(
            "{}: scoring contains unknown keys: {}".format(source, ", ".join(map(str, unknown)))
        )
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("{}: scoring.version must be an integer".format(source))
    if version not in SUPPORTED_ROUTING_SCORING_VERSIONS:
        raise ValueError(
            "{}: unsupported scoring.version {}; supported: {}".format(
                source, version, ", ".join(map(str, SUPPORTED_ROUTING_SCORING_VERSIONS))
            )
        )
    if kind not in ROUTING_SCORING_TASK_KINDS:
        raise ValueError(
            "{}: scoring versions are defined only for routing source kinds {}".format(
                source, ", ".join(sorted(ROUTING_SCORING_TASK_KINDS))
            )
        )
    if version == 1:
        if "overflow_gate" in value:
            raise ValueError("{}: scoring.overflow_gate requires scoring.version 2".format(source))
        return {}
    gate = value.get("overflow_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("{}: scoring version 2 requires an overflow_gate object".format(source))
    unknown = sorted(set(gate) - {"max_ratio", "abs_slack"})
    if unknown:
        raise ValueError(
            "{}: scoring.overflow_gate contains unknown keys: {}".format(
                source, ", ".join(map(str, unknown))
            )
        )
    if "abs_slack" not in gate:
        raise ValueError("{}: scoring.overflow_gate.abs_slack is required".format(source))
    return {
        "version": 2,
        "overflow_gate": {
            "max_ratio": _finite_non_negative(
                gate.get("max_ratio", DEFAULT_OVERFLOW_GATE_MAX_RATIO),
                "scoring.overflow_gate.max_ratio",
            ),
            "abs_slack": _finite_non_negative(
                gate["abs_slack"], "scoring.overflow_gate.abs_slack"
            ),
        },
    }


def scoring_version(scoring: Optional[Mapping[str, Any]]) -> int:
    if not scoring:
        return 1
    return int(scoring.get("version", 1))


def overflow_gate(
    total_overflow: float,
    reference_overflow: float,
    scoring: Mapping[str, Any],
) -> Tuple[bool, float, str]:
    """Return ``(passed, limit, reason)`` for one version-2 overflow gate."""

    gate = scoring["overflow_gate"]
    max_ratio = float(gate["max_ratio"])
    abs_slack = float(gate["abs_slack"])
    limit = max_ratio * reference_overflow + abs_slack
    passed = total_overflow <= limit
    reason = (
        "candidate total GRT overflow {cand:g} {op} limit {limit:g} "
        "= max_ratio {ratio:g} * reference overflow {ref:g} + abs_slack {slack:g}"
    ).format(
        cand=total_overflow,
        op="<=" if passed else ">",
        limit=limit,
        ratio=max_ratio,
        ref=reference_overflow,
        slack=abs_slack,
    )
    if not passed:
        reason = "overflow gate failed: " + reason + "; combined_score is 0"
    else:
        reason = "overflow gate passed: " + reason
    return passed, limit, reason


def scoring_contract_text(scoring: Mapping[str, Any]) -> str:
    """Prompt section that overrides the version-1 formula in task prompts."""

    gate = scoring["overflow_gate"]
    return (
        "## Scoring contract override (routing scoring version 2)\n\n"
        "This task instance uses routing scoring version 2, which replaces any "
        "version-1 blended overflow/wirelength formula stated above. Let O and W "
        "be the candidate total (bottom + top) GRT overflow and wire length, and "
        "O_ref and W_ref the same-contract reference values.\n\n"
        "- Overflow hard gate: the candidate passes iff "
        "O <= {ratio:g} * O_ref + {slack:g}. A candidate that fails the gate "
        "scores combined_score = 0 and the feedback reports the gate reason.\n"
        "- When the gate passes: combined_score = grt_complete * W_ref / W "
        "(higher is better; overflow below the gate earns no extra credit).\n"
        "- Without a reference run (bootstrap only) the gate is not evaluated and "
        "combined_score = grt_complete / (1 + W / 1e6).\n"
    ).format(ratio=float(gate["max_ratio"]), slack=float(gate["abs_slack"]))
