"""Distinguish historical TP-GNN from the current, explicitly flat contract.

Configuration aliases are user conveniences, not evidence that two frozen
algorithm baselines are interchangeable. Callers retain their hash checks.
"""
from typing import Mapping, Optional


TP_GNN_METHODS = frozenset(("tp-gnn", "tp-gnn-flat"))


def validate_tp_gnn_method(report: Mapping, *recorded_methods: str,
                           expected_method: Optional[str] = None) -> str:
    method = report.get("method")
    if method not in TP_GNN_METHODS:
        raise ValueError("partition method report is not TP-GNN; actual TP-GNN provenance is required")
    if any(recorded != method for recorded in recorded_methods):
        raise ValueError("recorded partition method is not TP-GNN matching its native report; identities disagree, do not relabel artifacts")
    hierarchy = report.get("hierarchy", {})
    if method == "tp-gnn-flat":
        config = report.get("config")
        if (not isinstance(hierarchy, Mapping) or hierarchy.get("mode") != "flat"
                or hierarchy.get("consumed_as_feature") is not False
                or not isinstance(config, Mapping) or not config.get("decoder")):
            raise ValueError("tp-gnn-flat requires native flat-hierarchy evidence and decoder config")
    elif isinstance(hierarchy, Mapping) and hierarchy.get("mode") == "flat":
        raise ValueError("historical tp-gnn cannot be relabeled as the flat algorithm")
    if expected_method is not None and method != expected_method:
        raise ValueError("TP-GNN algorithm contract differs from frozen reference/profile ({} != {}); qualify a new profile".format(method, expected_method))
    return str(method)
