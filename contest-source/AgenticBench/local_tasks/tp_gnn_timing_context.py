"""TP-GNN timing-context-v2 adapter; frozen training recipe and scoring."""
from __future__ import annotations

from pathlib import Path

import ast
import inspect
import math
from local_tasks.tp_gnn_loss_patch import (
    _patch_tp_gnn as _legacy_patch, _replace_once, TP_GNN,
    _validate_objective, OBJECTIVE_ARGS as LEGACY_ARGS,
)
from local_bench.program_safety import read_repo_file, single_definition, unified_diff
from local_bench.numerical_programs import validate_local_reads


OBJECTIVE_ARGS = LEGACY_ARGS + [
    "graph_scores", "graph_sources", "graph_targets", "graph_net_ids",
    "graph_weights", "net_weights", "node_features", "node_areas",
]


def _agentic_loss_graph(groups, nets, weights, areas, feature_regime):
    """Complete contracted hypergraph as deterministic linear-size stars.

    Each net's smallest group ID is an arbitrary anchor, NOT its driver.
    Duplicate pins/groups are removed; parallel nets remain distinct. No
    max-degree truncation, timing recomputation, RNG or training changes.
    """
    if feature_regime != "partition-last" or weights is None or len(weights) != len(nets):
        raise ValueError("Timing loss context requires complete frozen partition-last weights")
    owner = {name: i for i, group in enumerate(groups) for name in group}
    if len(owner) != sum(len(group) for group in groups):
        raise ValueError("Loss context contains duplicate group membership")
    if len(areas) != len(groups) or any(not math.isfinite(a) or a <= 0 for a in areas):
        raise ValueError("Loss context requires finite positive group areas")
    sources, targets, net_ids, pair_weights = [], [], [], []
    for net_id, pins in enumerate(nets):
        weight = weights[net_id]
        if not math.isfinite(weight) or weight < 1:
            raise ValueError("Loss context has invalid frozen net weight")
        if any(pin not in owner for pin in pins):
            raise ValueError("Loss context net contains an unknown instance")
        members = sorted({owner[pin] for pin in pins})
        for target in members[1:]:
            sources.append(members[0])
            targets.append(target)
            net_ids.append(net_id)
            pair_weights.append(weight / (len(members) - 1))
    return sources, targets, net_ids, pair_weights, list(weights), list(areas)


def _patch_tp_gnn(before: str, objective: str) -> str:
    native_hook = "    loss_context=None,\n    evolution=None,\n" in before
    call_indent = "                " if native_hook else "            "
    after = _legacy_patch(before, objective)
    after = _replace_once(after,
        '            "loss": "OpenAgentic3DBench-evolved compute_tp_gnn_loss",\n',
        '            "loss": "OpenAgentic3DBench-evolved compute_tp_gnn_loss",\n'
        '            "loss_context_interface": "timing-context-v2",\n',
        "training context config provenance")
    after = _replace_once(after,
        '        "loss_semantics": "OpenAgentic3DBench-evolved objective over frozen positive/negative samples",\n',
        '        "loss_semantics": "OpenAgentic3DBench-evolved objective over frozen positive/negative samples",\n'
        '        "loss_context_interface": "timing-context-v2",\n'
        '        "loss_context_star_edges": len(loss_context[0]),\n'
        '        "loss_context_hyperedges": len(loss_context[4]),\n',
        "training context report provenance")
    after = _replace_once(after, "def train_random_walk_embeddings(\n",
        inspect.getsource(_agentic_loss_graph) + "\n\ndef train_random_walk_embeddings(\n",
        "sparse timing context builder")
    if native_hook:
        # The optional context parameter already exists. Feed the old task's
        # context only to the inactive-capsule call, leaving native evolution
        # dispatch and its separately verified context unchanged.
        after = _replace_once(after,
            '                features, hierarchy_ids, feature_report["known_hierarchy_count"],\n'
            '                group_neighbors, config)\n',
            '                features, hierarchy_ids, feature_report["known_hierarchy_count"],\n'
            '                group_neighbors, config,\n'
            '                _agentic_loss_graph(groups, nets, timing_net_weights, group_areas, config.feature_regime))\n',
            "inactive native training context handoff")
    else:
        after = _replace_once(after,
            "    config: TPGNNConfig,\n) -> tuple[list[list[float]], dict]:\n",
            "    config: TPGNNConfig,\n    loss_context,\n) -> tuple[list[list[float]], dict]:\n",
            "training context interface")
        after = _replace_once(after,
            '            feature_report["known_hierarchy_count"],\n            group_neighbors,\n            config,\n',
            '            feature_report["known_hierarchy_count"],\n            group_neighbors,\n            config,\n'
            '            _agentic_loss_graph(groups, nets, timing_net_weights, group_areas, config.feature_regime),\n',
            "training context handoff")
    setup = '''    graph_plan = deterministic_pair_plan(loss_context[0], loss_context[1])
    graph_sources, graph_targets = graph_plan[:2]
    graph_net_ids = torch.tensor(loss_context[2], dtype=torch.long, device=device)
    graph_weights = torch.tensor(loss_context[3], dtype=x.dtype, device=device)
    net_weights = torch.tensor(loss_context[4], dtype=x.dtype, device=device)
    node_features = torch.tensor(features, dtype=x.dtype, device=device)
    node_areas = torch.tensor(loss_context[5], dtype=x.dtype, device=device)
    if node_features.shape != (n_nodes, 10):
        raise ValueError("Loss context requires ten partition-last feature columns")
    if not all(bool(torch.isfinite(t).all().item()) for t in
               (graph_weights, net_weights, node_features, node_areas)):
        raise ValueError("Non-finite loss context")

'''
    after = _replace_once(after, "    for epoch in range(effective_epochs):\n",
        setup + "    for epoch in range(effective_epochs):\n", "timing tensor setup")
    after = _replace_once(after, call_indent + "loss = compute_tp_gnn_loss(\n",
        call_indent + "graph_scores = deterministic_pair_dot(emb, graph_plan)\n"
        + call_indent + "loss = compute_tp_gnn_loss(\n", "full graph differentiable scores")
    argument_indent = call_indent + "    "
    after = _replace_once(after, argument_indent + "len(source_batch), epoch, effective_epochs,\n",
        argument_indent + "len(source_batch), epoch, effective_epochs,\n"
        + argument_indent + "graph_scores, graph_sources, graph_targets, graph_net_ids,\n"
        + argument_indent + "graph_weights, net_weights, node_features, node_areas,\n",
        "timing loss arguments")
    return _epoch_audit_log(before, after)


def _epoch_audit_log(before: str, after: str) -> str:
    """Restore audit-only epoch events for producers that only publish CSV.

    This runs after optimizer updates and CSV-row construction. It reads only
    existing scalar values; no RNG, tensors, convergence or counters change.
    The old producer's native event is preserved without duplication.
    """
    conditional_event = ('        if evolved_loss is not None:\n'
        '            print(f"[tp-gnn] epoch {epoch + 1}/{effective_epochs}: loss={avg_loss:.8f} batches={total_batches} optimizer_steps={optimizer_steps_completed}", flush=True)\n')
    if conditional_event in before:
        # Native capsule-only logging is otherwise silent for a historical
        # source-patched task. Make that same audit event unconditional.
        return _replace_once(after, conditional_event,
            conditional_event.split("\n", 1)[1][4:], "native inactive-branch epoch audit")
    if '[tp-gnn] epoch ' in before:
        return after
    anchor = '        log_rows.append({"epoch": epoch + 1, "loss": avg_loss})\n'
    event = ('        print(f"[tp-gnn] epoch {epoch + 1}/{effective_epochs}: "\n'
             '              f"loss={avg_loss:.8f} batches={total_batches} "\n'
             '              f"optimizer_steps={optimizer_steps_completed}", flush=True)\n')
    return _replace_once(after, anchor, anchor + event, 'trusted epoch audit event')


def materialize(program_text: str, open3dbench_root: Path) -> dict:
    """Retain the standard loss safety/patch contract with an explicit flat ID.

    Training, global-2way decoding and frozen STA inputs remain evaluator-owned.
    Reusing the existing parser avoids a weaker parallel AST safety policy.
    """
    before = read_repo_file(open3dbench_root, TP_GNN)
    objective, node = single_definition(program_text, ast.FunctionDef, 'compute_tp_gnn_loss')
    _validate_objective(node, OBJECTIVE_ARGS)
    if any(isinstance(n, (ast.AugAssign, ast.Try)) for n in ast.walk(node)):
        raise ValueError("Loss forbids augmented assignment and exception handling")
    # A nested hook must not read its enclosing optimizer/model/config scope.
    helpers = {"torch", "F", "abs", "float", "int", "len", "max", "min", "range"}
    validate_local_reads(node, helpers, protected=helpers)
    after = _patch_tp_gnn(before, objective)
    candidate = {"schema_version": 1, "candidate_files": {},
                 "source_patch": unified_diff(TP_GNN, before, after)}
    candidate["config_overrides"] = {"partition": {"algorithm": "tp-gnn-flat"}}
    candidate["notes"] = (
        "TP-GNN-flat timing-context-v2 loss integrated by the trusted adapter; "
        "fixed global-2way recipe and frozen timing weights unchanged. "
        "Native reference authority is not established by materialization."
    )
    return candidate
