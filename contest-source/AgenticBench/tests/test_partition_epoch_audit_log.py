import ast
import pytest
from local_tasks.tp_gnn_timing_context import _epoch_audit_log


def test_audit_event_leaves_computation_ast_identical(capsys):
    before = ('def train():\n'
              '    for epoch in range(1):\n'
              '        log_rows.append({"epoch": epoch + 1, "loss": avg_loss})\n')
    after = _epoch_audit_log(before, before)
    tree = ast.parse(after)
    inserted = tree.body[0].body[0].body.pop()
    assert ast.dump(tree) == ast.dump(ast.parse(before))
    assert isinstance(inserted, ast.Expr) and inserted.value.func.id == 'print'
    namespace = dict(log_rows=[], avg_loss=40.73516082763672, effective_epochs=1,
                     total_batches=6, optimizer_steps_completed=6)
    exec(compile(after, '<trusted audit fixture>', 'exec'), namespace)
    namespace['train']()
    assert capsys.readouterr().out == '[tp-gnn] epoch 1/1: loss=40.73516083 batches=6 optimizer_steps=6\n'
    assert namespace['log_rows'] == [dict(epoch=1, loss=40.73516082763672)]


def test_old_native_log_not_duplicated_and_unknown_source_rejected():
    old = 'print("[tp-gnn] epoch ")'
    assert _epoch_audit_log(old, old) == old
    with pytest.raises(ValueError):
        _epoch_audit_log('unknown producer', 'unknown producer')
