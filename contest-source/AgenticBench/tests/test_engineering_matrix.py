import json
from unittest.mock import patch

import pytest

from engineering_bench.matrix import run
from grt_bench.queue import Queue
from grt_bench.runtime import UncertainExecution


def plan(root):
    cell = root / 'cell'
    (cell / 'experiment').mkdir(parents=True)
    receipt = cell / 'experiment/preparation-result.json'
    receipt.write_text(json.dumps({'status': 'qualified'}))
    data = {'capacity': {'cpu': 8, 'memory_gb': 32, 'storage_gb': 40, 'gpus': [], 'build_slots': 1},
            'resources': {'cpu': 8, 'memory_gb': 32, 'storage_gb': 20, 'gpu_count': 0, 'build_slots': 0},
            'cells': [{'case': 'bp_fe', 'seed': 's1', 'root': str(cell)}]}
    (root / 'plan.json').write_text(json.dumps(data))


def test_matrix_uses_real_lease_and_does_not_retry_finished_cell(tmp_path):
    plan(tmp_path)
    with patch('engineering_bench.matrix.evaluator_identity', return_value='frozen'), \
         patch('engineering_bench.matrix.subprocess.Popen') as process, \
         patch('engineering_bench.matrix.subprocess.check_output', return_value=''):
        process.return_value.wait.return_value = 0
        run(tmp_path, 'prepare', ['bp_fe'], ['s1'], 1)
        assert process.call_count == 1
        assert process.call_args.kwargs['start_new_session'] is True
        row = Queue(tmp_path / 'control/engineering-queue.sqlite').rows()[0]
        assert row['state'] == 'succeeded'
        assert json.loads(row['resources'])['gpu_count'] == 0
        with pytest.raises(FileExistsError):
            run(tmp_path, 'prepare', ['bp_fe'], ['s1'], 1)
        assert process.call_count == 1


def test_matrix_retains_resources_when_container_cleanup_uncertain(tmp_path):
    plan(tmp_path)
    with patch('engineering_bench.matrix.evaluator_identity', return_value='frozen'), \
         patch('engineering_bench.matrix.subprocess.Popen') as process, \
         patch('engineering_bench.matrix.subprocess.check_output', return_value='remaining\n'), \
         patch('engineering_bench.matrix.stop_and_confirm', side_effect=UncertainExecution('unknown')):
        process.return_value.wait.return_value = 0
        with pytest.raises(UncertainExecution):
            run(tmp_path, 'prepare', ['bp_fe'], ['s1'], 1)
    row = Queue(tmp_path / 'control/engineering-queue.sqlite').rows()[0]
    assert row['state'] == 'running' and row['expires'] is not None
    assert Queue(tmp_path / 'control/engineering-queue.sqlite').claim('rafa1-engineering') is None
