"""Validate an externally collected, complete final-only reviewer bundle."""


def validate_bundle(bundle, task, sealed):
    if bundle.get('kind') != 'engineering-judge-legality-bundle-v1' or bundle.get('task') != task.task_id:
        raise ValueError('Wrong judge evidence bundle')
    if bundle.get('submission_sha256') != sealed['submission_sha256'] or bundle.get('task_digest') != task.digest():
        raise ValueError('Reviewer evidence is not bound to the sealed task/submission')
    rows = bundle.get('cases', [])
    if len(rows) != len(task.data['cases']) or {r['case'] for r in rows} != set(task.data['cases']):
        raise ValueError('Reviewer needs each case exactly once')
    for row in rows:
        if row['submission_sha256'] != sealed['submission_sha256'] or row['task_digest'] != task.digest():
            raise ValueError('Case evidence identity mismatch')
        if row['status'] not in ('succeeded', 'failed') or type(row.get('legal')) is not bool:
            raise ValueError('Do not grade unfinished/unknown cases as final outcomes')
        if row['status'] == 'succeeded' and (not row['legal'] or not row['checks'] or not all(v is True for v in row['checks'].values())):
            raise ValueError('Missing successful native checks')
        for banned in ('candidate_metric', 'reference_metric', 'score', 'judge_score', 'model', 'developer_model'):
            if banned in row:
                raise ValueError('Ranking/identity leakage into reviewer evidence')
    # Keep detailed host provenance outside the model workspace. Native receipt
    # hashes stay available to trace the evidence without leaking prior scores.
    return [{k: v for k, v in row.items() if k not in ('provenance', 'submission_sha256', 'task_digest')}
        | {'native_receipt_sha256': row['provenance'].get('native_sha256'),
           'ledger_sha256': row['provenance']['ledger_sha256']} for row in rows]
