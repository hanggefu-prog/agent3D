from copy import deepcopy
from types import SimpleNamespace
import pytest
from judge_bundle import validate_bundle


def sample():
    task = SimpleNamespace(task_id='x', data={'cases': ['a', 'b']}, digest=lambda: 'task-hash')
    sealed = {'submission_sha256': 'submission-hash'}
    rows = [{'case': case, 'submission_sha256': 'submission-hash', 'task_digest': 'task-hash',
        'status': 'succeeded', 'legal': True, 'checks': {'physical': True},
        'provenance': {'ledger_sha256': 'receipt'}} for case in task.data['cases']]
    bundle = {'kind': 'engineering-judge-legality-bundle-v1', 'task': 'x',
        'submission_sha256': 'submission-hash', 'task_digest': 'task-hash', 'cases': rows}
    return bundle, task, sealed


def test_complete_bundle_and_redaction():
    bundle, task, sealed = sample()
    result = validate_bundle(bundle, task, sealed)
    assert [r['case'] for r in result] == ['a', 'b']
    assert not any('provenance' in r for r in result)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'identity', 'pending', 'score', 'checks'])
def test_incomplete_or_biased_bundle_refused(change):
    bundle, task, sealed = sample()
    if change == 'missing': bundle['cases'].pop()
    if change == 'duplicate': bundle['cases'][1] = deepcopy(bundle['cases'][0])
    if change == 'identity': bundle['cases'][1]['submission_sha256'] = 'other'
    if change == 'pending': bundle['cases'][1]['status'] = 'dispatched'
    if change == 'score': bundle['cases'][1]['score'] = 5
    if change == 'checks': bundle['cases'][1]['checks'] = {}
    with pytest.raises(ValueError): validate_bundle(bundle, task, sealed)
