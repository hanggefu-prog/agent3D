"""Run a reviewed campaign; finished candidate failures spend once and continue.

The authorization binds this driver, configuration, task, model and prompt sources.
It is host-authored, never an agent tool. Starting twice requires reconciliation.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from grt_bench.common import digest,read,write,sha
from .grt_campaign import Campaign
from .grt_retention_lifecycle import after_round


def round_disposition(row):
    status=row['status']
    if status in {'verified','infeasible'}:
        return 'evaluated'
    if status=='failed' and row.get('evaluation_failure'):
        return 'evaluated'
    if status in {'no_change','blocked','rejected','generation_failed'}:
        return 'unsubmitted'
    raise ValueError('Uncertain round state requires reconciliation: '+status)


def run(authorization_path):
    path=Path(authorization_path).resolve(strict=True);auth=read(path)
    campaign=Campaign(auth['config']);folder=path.parent
    def guard():
        expected={'config_digest':digest(campaign.config),'task_digest':campaign.task_id,
            'implementation_digest':campaign.implementation_digest(),'driver_sha256':sha(__file__),
            'model':campaign.config['model'],'evaluation_limit':campaign.config['evaluation_limit']}
        if any(auth.get(k)!=v for k,v in expected.items()) or read(campaign.config_path)!=campaign.config:
            raise ValueError('Campaign differs from reviewed authorization')
        if auth.get('retention_sha256') and sha(folder/'retention.py') != auth['retention_sha256']:
            raise ValueError('Campaign retention helper changed')
    def status(phase,**kwargs):
        event=dict(phase=phase,at=time.time(),pid=os.getpid(),**kwargs)
        write(folder/'status.json',event,replace=True)
        with (folder/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
    with (folder/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
        if (folder/'started.json').exists():
            raise ValueError('Reconcile an existing launch; never replay it automatically')
        write(folder/'started.json',dict(at=time.time(),pid=os.getpid(),authorization_sha256=sha(path)))
        failures=0
        try:
            while read(campaign.state_path)['evaluations_used']<auth['evaluation_limit']:
                guard()
                initial_retention = after_round(campaign.root, campaign.task)
                if initial_retention is not None:
                    status('retention_completed', receipt=initial_retention['receipt'],
                           evaluations_used=read(campaign.state_path)['evaluations_used'])
                if shutil.disk_usage(campaign.root).free < auth.get('minimum_free_gib',150)*1024**3:
                    raise RuntimeError('Insufficient free storage before next round')
                row=campaign.prepare()
                if row['round']==1 and row['prompt_sha256']!=auth['first_prompt_sha256']:
                    raise ValueError('First prompt differs from reviewed text')
                write(Path(row['control'])/'user-authorization.json',{
                    'authorization_sha256':sha(path),'prompt_sha256':row['prompt_sha256']})
                status('round_running',round=row['round'])
                result=campaign.run(row['prompt_sha256'])
                kind=round_disposition(result)
                retention_destination = None
                if result['status'] in {'verified','infeasible'} and auth.get('retention_sha256'):
                    spec=importlib.util.spec_from_file_location('reviewed_retention',folder/'retention.py')
                    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
                    retention_destination = folder/'retention'/('%04d' % row['round'])
                    helper.compact_matrix(campaign,result,retention_destination)
                retained = after_round(campaign.root, campaign.task, result,
                                       legacy_retention=retention_destination)
                if retained is not None:
                    status('retention_completed', round=row['round'], receipt=retained['receipt'],
                           evaluations_used=read(campaign.state_path)['evaluations_used'])
                if 'unexpected model identity' in result.get('history',{}).get('feedback',''):
                    raise RuntimeError(result['history']['feedback'])
                failures=failures+1 if kind=='unsubmitted' else 0
                status('round_finished',round=row['round'],status=result['status'],
                       candidate=result.get('candidate'),evaluations_used=read(campaign.state_path)['evaluations_used'])
                if failures>=auth['consecutive_pre_submission_failure_limit']:
                    raise RuntimeError('Consecutive pre-submission failure limit reached without spending evaluations')
            if campaign.config.get('final_test_policy')=='baseline-and-final-training-frontier':
                from .grt_final_test import run_final_tests
                status('final_tests_running')
                run_final_tests(campaign.config,campaign.task)
            status('completed',evaluations_used=read(campaign.state_path)['evaluations_used'])
        except BaseException as error:
            status('stopped_with_error',error_type=type(error).__name__,error=str(error))
            raise


if __name__=='__main__':run(sys.argv[1])
