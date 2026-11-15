"""Failure/restart contract of the shared outer model loop."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from contest2.budget import BudgetLedger, BudgetExceeded
from contest2.model_pricing import reserve_full_context, MODEL_ID
from contest2.runtime_recovery import lease, recoverable_turn, recover_provider, provider_directory
from grt_bench.common import digest, read, write


def ledger(tmp_path):
    return BudgetLedger(tmp_path / 'budget.sqlite', evaluation_limit=10,
                        model_cost_limit_rmb='15', task_id='binary_loss', allowed_model=MODEL_ID)


def test_insufficient_next_model_reservation_does_not_retry_or_block_evaluation(tmp_path):
    book = ledger(tmp_path)
    for i in range(7):
        book.reserve_model_call(str(i), MODEL_ID, reserve_full_context(peak=True))
        book.record_model_call(str(i), MODEL_ID, 1, 1, '2')
    state = book.snapshot()
    assert state['model_cost_available_rmb'] == '1'
    assert not state['dispatch_allowed'] and not state['model_dispatch_allowed']
    assert state['evaluation_dispatch_allowed']
    assert 'insufficient_model_reservation' in state['model_stop_reasons']
    with pytest.raises(BudgetExceeded):
        book.reserve_model_call('next', MODEL_ID, reserve_full_context(peak=True))
    book.accept_evaluation('eval', 'binary_loss', 'a' * 64)


def test_uncertain_bill_keeps_reservation_but_cannot_deadlock_final_freeze(tmp_path):
    book = ledger(tmp_path)
    book.accept_evaluation('eval', 'binary_loss', 'a' * 64)
    book.finish_evaluation('eval', 'succeeded')
    book.reserve_model_call('lost', MODEL_ID, '2')
    with pytest.raises(ValueError, match='in-flight'):
        book.freeze('a' * 64, 'b' * 64)
    book.mark_model_uncertain('lost', 'transport ended without usage')
    assert book.snapshot()['model_cost_reserved_rmb'] == '2'
    assert book.snapshot()['model_cost_spent_rmb'] == '0'
    book.freeze('a' * 64, 'b' * 64)
    book.record_model_call('lost', MODEL_ID, 100, 20, '0.1')
    assert book.snapshot()['model_cost_reserved_rmb'] == '0'
    assert book.snapshot()['model_cost_spent_rmb'] == '0.1'


def test_gateway_restart_retains_unknown_usage_and_recovers_durable_receipts(tmp_path):
    book = ledger(tmp_path)
    for request in ('a' * 32, 'b' * 32):
        with lease(provider_directory(book) / (request + '.lock')):
            book.reserve_model_call(request, MODEL_ID, '2')
    with lease(provider_directory(book) / ('b' * 32 + '.lock')):
        recover_provider(book)
        rows = {r['request_id']: r for r in book.snapshot()['model_calls']}
        assert rows['a' * 32]['status'] == 'uncertain'
        assert rows['b' * 32]['status'] == 'reserved'
    recover_provider(book)
    assert all(r['status'] == 'uncertain' for r in book.snapshot()['model_calls'])


def test_interrupted_turn_recovers_without_replaying_the_old_request(tmp_path):
    class Controller:
        codex_directory = tmp_path
        calls = 0
        @recoverable_turn
        def turn(self, prompt, request_id):
            self.calls += 1
            return {'success': True}
    write(tmp_path / 'state.json', {'thread_id': None, 'turns': {
        'old': {'status': 'pending', 'prompt_digest': digest('edit'), 'ownership': 'inherited-process-lock'}}})
    ctl = Controller()
    recovered = ctl.turn('edit', 'old')
    assert recovered['status'] == 'interrupted' and ctl.calls == 0
    with lease(tmp_path / 'turn.lock'):
        assert ctl.turn('next edit', 'next')['status'] == 'running'
        assert ctl.calls == 0
    assert ctl.turn('next edit', 'next')['success'] and ctl.calls == 1


def test_turn_lock_survives_controller_descriptor_close(tmp_path):
    lock = tmp_path / 'turn.lock'
    with lease(lock) as descriptor:
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, pass_fds=(descriptor,))
    try:
        with pytest.raises(BlockingIOError):
            with lease(lock, nonblocking=True):
                pass
    finally:
        child.communicate(b'')
    with lease(lock, nonblocking=True):
        pass


def test_phased_adapter_binds_unlimited_executor_and_preserves_physical_source():
    import hashlib
    from contest2.execution_policy import install
    from local_bench import evaluation
    from local_bench.phased import split_command
    before = hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest()
    install()
    assert evaluation.evaluate_task.__globals__['_execute_prepared'] is evaluation._execute_prepared
    assert hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest() == before
    command = ['python', '-m', 'open3dbench', 'run', 'config', '--from', 'partition', '--to', 'opt', '--resume']
    front, back = split_command(command, 'frontend'), split_command(command, 'backend')
    assert front[front.index('--to')+1] == 'placement'
    assert back[back.index('--from')+1] == 'pdn' and back[back.index('--to')+1] == 'opt'


def test_wrong_baseline_is_checked_before_irreversible_freeze(tmp_path, monkeypatch):
    from contest2 import runner, halo_policy, sampling
    from local_tasks.tp_gnn_local_task_definitions import materialize_binary_loss
    root = tmp_path / 'work'
    candidate = 'a' * 64
    source = (Path(runner.__file__).parents[1] / 'agent/binary_loss/program.py').read_text()
    write(root / 'candidates' / candidate / 'candidate.json', materialize_binary_loss(source))
    book = BudgetLedger(root / 'control/budget.sqlite', evaluation_limit=10,
                        model_cost_limit_rmb='15', task_id='binary_loss', allowed_model=MODEL_ID)
    book.accept_evaluation('eval', 'binary_loss', candidate)
    book.finish_evaluation('eval', 'succeeded')
    monkeypatch.setattr(runner, '_runtime', lambda *a, **kw: (object(), {}))
    monkeypatch.setattr(runner, 'agent_digest', lambda path: 'b' * 64)
    monkeypatch.setattr(runner, 'LocalEvaluationMatrix', lambda *a: SimpleNamespace(
        cases=['swerv_wrapper-ng45','bp_fe-asap7'], cells=[{}]*6,
        seeds=[{'partition': seed} for seed in (1001,1002,1003)]))
    monkeypatch.setattr(halo_policy, 'validate_matrix_halos', lambda *a: None)
    monkeypatch.setattr(sampling, 'validate_public_matrix', lambda *a: None)
    args = SimpleNamespace(work_root=root, source=tmp_path, candidate_id=candidate,
        command='validate', baseline='loss_decoder', matrix=tmp_path/'matrix', runtime=tmp_path/'runtime')
    with pytest.raises(ValueError, match='task_id'):
        runner.validate(args)
    assert book.snapshot()['frozen_candidate'] is None


def test_real_shim_exec_preserves_turn_ownership_and_session_receipt(tmp_path):
    from contest2 import codex_runtime, codex_shim
    control = tmp_path / 'control'
    control.mkdir()
    fake = tmp_path / 'fake-codex'
    fake.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
os.fstat(int(os.environ["CONTEST_TURN_LEASE_FD"]))
sys.stdin.read()
Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text("owned CLI process completed")
print(json.dumps({"type":"thread.started","thread_id":"11111111-1111-1111-1111-111111111111"}))
print(json.dumps({"type":"turn.completed"}))
''')
    fake.chmod(0o755)
    with lease(tmp_path/'turn.lock') as descriptor:
        result = codex_runtime.run_workspace_turn(binary=codex_shim.__file__, control=control,
            output=tmp_path/'turn', container='o3db-agent-test', prompt='edit',
            terminal_script=tmp_path/'terminal.py', python=sys.executable, model=MODEL_ID,
            environment={'PATH':str(Path(sys.executable).parent)+':/usr/bin:/bin',
                'CONTEST_CODEX_BINARY':str(fake), 'CONTEST_GATEWAY_URL':'http://127.0.0.1:12345/v1',
                'CONTEST_GATEWAY_TOKEN':'test-token-'*4, 'CONTEST_TURN_LEASE_FD':str(descriptor)})
    assert result['success'] and result['turn_completed']
    assert result['final_message'] == 'owned CLI process completed'
