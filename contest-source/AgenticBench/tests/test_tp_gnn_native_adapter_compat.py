"""Historical loss patches target only the inactive native capsule branch."""
import ast
import hashlib
import json
from pathlib import Path

from local_bench.program_safety import single_definition
from local_tasks.tp_gnn_timing_context import _patch_tp_gnn


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "AgenticBench/local_tasks/tp_gnn_post_placement_timing/initial_program.py"
NATIVE = ROOT / "Place-3D/src/partition/lol/methods/tp_gnn.py"



def assert_historical_initial_candidate(spec, candidate, expected_digest):
    """Keep frozen bytes while permitting only native source-context rebasing."""
    from local_bench.model import load_candidate
    from local_bench.program_safety import unified_diff
    from local_bench.verifier import verify_candidate
    frozen = load_candidate(spec.initial_candidate)
    assert hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() == expected_digest
    assert {key: value for key, value in candidate.items() if key != "source_patch"} == {
        key: value for key, value in frozen.items() if key != "source_patch"}
    assert hashlib.sha256(spec.program_template.read_bytes()).hexdigest() == (
        "ac7494028435724ffbd2ef2922d958131b17ef058e7e462658b0f304b39460ca")
    proof = verify_candidate(spec, candidate, ROOT)
    assert proof["patch"]["paths"] == ["Place-3D/src/partition/lol/methods/tp_gnn.py"]
    source, initial = single_definition(spec.program_template.read_text(), ast.FunctionDef, "compute_tp_gnn_loss")
    before = NATIVE.read_text()
    after = _patch_tp_gnn(before, source)
    assert candidate["source_patch"] == unified_diff("Place-3D/src/partition/lol/methods/tp_gnn.py", before, after)
    definitions = [node for node in ast.walk(ast.parse(after))
                   if isinstance(node, ast.FunctionDef) and node.name == "compute_tp_gnn_loss"]
    assert len(definitions) == 1
    assert ast.dump(definitions[0]) == ast.dump(initial)


def objective():
    return single_definition(PROGRAM.read_text(), ast.FunctionDef, "compute_tp_gnn_loss")[0]


def find_if(source, condition, containing):
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == containing)
    return next(node for node in ast.walk(function)
                if isinstance(node, ast.If) and ast.unparse(node.test) == condition)


def test_historical_patch_leaves_native_loss_and_decoder_capsule_branches_unchanged():
    before = NATIVE.read_text()
    after = _patch_tp_gnn(before, objective())
    ast.parse(after)
    old = find_if(before, "evolved_loss is None", "train_random_walk_embeddings")
    new = find_if(after, "evolved_loss is None", "train_random_walk_embeddings")
    assert [ast.dump(node) for node in old.orelse] == [ast.dump(node) for node in new.orelse]
    # The legacy training call receives the source-patched context; the native
    # capsule call and namespace remain exactly as the physical source defined.
    old = find_if(before, "evolution is None", "tp_gnn_partition")
    new = find_if(after, "evolution is None", "tp_gnn_partition")
    assert [ast.dump(node) for node in old.orelse] == [ast.dump(node) for node in new.orelse]
    assert after.count('print(f"[tp-gnn] epoch ') == 1
    function = next(node for node in ast.parse(after).body
                    if isinstance(node, ast.FunctionDef) and node.name == "train_random_walk_embeddings")
    assert [arg.arg for arg in function.args.args].count("loss_context") == 1
    assert [arg.arg for arg in function.args.args].count("evolution") == 1


def test_original_source_shape_still_materializes_without_native_hook_parameters():
    # A small original-shape producer exercises the immutable textual anchors
    # without borrowing any historical native receipt or rewriting its fixture.
    source = '''from __future__ import annotations
def train_random_walk_embeddings(
    features,
    hierarchy_ids,
    hierarchy_count,
    neighbors,
    config: TPGNNConfig,
) -> tuple[list[list[float]], dict]:
    n_nodes = len(features)
    for epoch in range(effective_epochs):
        for source_batch in batches:
            loss = -(
                F.logsigmoid(pos_score).sum() + F.logsigmoid(-neg_score).sum()
            ) / len(source_batch)
        log_rows.append({"epoch": epoch + 1, "loss": avg_loss})
        training_config = {
            "loss": "per-source sum of positive-context and negative-sample log-sigmoid terms",
        }
    return embeddings, {
        "loss_semantics": "per-source sum over deduplicated contexts and M legal negatives",
    }

def partition():
    if enabled:
        embeddings, report = train_random_walk_embeddings(
            features,
            hierarchy_ids,
            feature_report["known_hierarchy_count"],
            group_neighbors,
            config,
        )
'''
    after = _patch_tp_gnn(source, objective())
    compile(after, "<original-shape-loss-patch>", "exec")
    tree = ast.parse(after)
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "train_random_walk_embeddings")
    assert [arg.arg for arg in trainer.args.args][-1] == "loss_context"
    assert "evolution" not in [arg.arg for arg in trainer.args.args]
    nested = next(node for node in trainer.body
                  if isinstance(node, ast.FunctionDef) and node.name == "compute_tp_gnn_loss")
    assert len(nested.args.args) == 18
    assert after.count('print(f"[tp-gnn] epoch ') == 1
