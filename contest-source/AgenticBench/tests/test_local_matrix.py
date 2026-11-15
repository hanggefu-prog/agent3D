"""Matrix completeness, paired references, seed delivery and ask/tell compatibility."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import threading
import time

import pytest
import yaml

from grt_bench.common import digest, sha, write
from local_bench.evaluation import _write_resolved_config, UPSTREAM_STAGE_DIRS
from local_bench.matrix import LocalEvaluationMatrix, qualify_matrix
from local_bench.model import EvaluationResult, load_task
from local_bench.program_bridge import materialize_program
from local_bench.search_api import CandidateProposal, TaskEvaluationOptions, ProgramEvaluationHarness
from local_bench.search_session import prepare_search_problem, run_task_search
from local_bench.seeds import validate_schedule, cell_contract, validate_executed_seeds
from local_bench.task_inventory import MATRIX_TASK_DIRECTORIES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent
SEEDS = [dict(id='s1', io=1, partition=1001, placement=1001, routing=-1),
         dict(id='s2', io=1, partition=1002, placement=1002, routing=-1)]


def fixture(tmp, name='placement_initialization', qualified=True):
    spec = load_task(ROOT / 'local_tasks' / name)
    data = dict(schema_version=1, task=spec.task_id, cases=['bp_fe-ng45', 'bp-ng45'], seeds=deepcopy(SEEDS), cells=[])
    for case in data['cases']:
        for seed in SEEDS:
            folder = tmp / case / seed['id']
            folder.mkdir(parents=True)
            baseline, reference = folder / 'input', folder / 'ref/run'
            baseline.mkdir(); reference.mkdir(parents=True)
            config = yaml.safe_load(spec.base_config.read_text())
            config['design'] = case.split('-')[0]
            for stage, key in [('pins', 'io'), ('partition', 'partition'), ('placement', 'placement'), ('routing', 'routing')]:
                config.setdefault(stage, {})['seed'] = seed[key]
            base = folder / 'base.yaml'
            base.write_text(yaml.safe_dump(config))
            for stage in spec.upstream_stages:
                write(baseline / UPSTREAM_STAGE_DIRS[stage] / 'stage_manifest.json', {'stage': stage})
            contract = {'task': spec.task_id, 'seeds': {k: v for k, v in seed.items() if k != 'id'},
                        'base_config_sha256': sha(base), 'upstream': {s: sha(baseline / UPSTREAM_STAGE_DIRS[s] /
                        'stage_manifest.json') for s in spec.upstream_stages}}
            write(reference.parent / 'evaluation_result.json', {'metrics': {'runs_successfully': 1, 'grt_complete': 1},
                'artifacts': {'matrix_reference': True, 'matrix_cell_contract': contract}})
            data['cells'].append(dict(case=case, seed=seed['id'], base_config=str(base),
                                      baseline_run=str(baseline), reference_run=str(reference) if qualified else None))
    path = tmp / 'matrix.json'; write(path, data)
    return spec, path, data


class Transport:
    def __init__(self, fail=None):
        self.preparations = 0; self.calls = []; self.active = 0; self.peak = 0
        self.lock = threading.Lock(); self.fail = fail
    def prepare_candidate(self, *args):
        self.preparations += 1
        return self
    def __call__(self, task, candidate, source, output, **kwargs):
        key = (Path(kwargs['base_config']).parent.parent.name, kwargs['seed_schedule']['placement'])
        with self.lock:
            self.calls.append(kwargs); self.active += 1; self.peak = max(self.peak, self.active)
        try:
            time.sleep(.01)
            if key == self.fail:
                raise RuntimeError('injected native failure')
            score = {('bp_fe-ng45', 1001): .8, ('bp_fe-ng45', 1002): 1.2,
                     ('bp-ng45', 1001): 1.1, ('bp-ng45', 1002): .9}[key]
            return EvaluationResult({'combined_score': score, 'runs_successfully': 1,
                                     'placement_native_hpwl': score * 100}, {})
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.parametrize('name', MATRIX_TASK_DIRECTORIES)
def test_all_eight_tasks_dispatch_complete_paired_matrices(tmp_path, name):
    spec, path, _ = fixture(tmp_path, name)
    transport = Transport()
    matrix = LocalEvaluationMatrix(path, spec)
    result = matrix.evaluate(tmp_path, SOURCE, tmp_path / 'out', transport=transport, workers=2)
    assert result.metrics['combined_score'] == pytest.approx(1)
    assert transport.preparations == 1 and len(transport.calls) == 4 and transport.peak == 2
    assert result.metrics['matrix']['cases'][0]['stddev'] == pytest.approx(.2)
    assert {r['seed_schedule']['placement'] for r in transport.calls} == {1001, 1002}
    assert all(r['reference_run'] is not None for r in transport.calls)


def test_failure_invalidates_matrix_but_keeps_other_cell_diagnostics(tmp_path):
    spec, path, _ = fixture(tmp_path)
    transport = Transport(fail=('bp-ng45', 1002))
    result = LocalEvaluationMatrix(path, spec).evaluate(tmp_path, SOURCE, tmp_path / 'out', transport=transport, workers=2)
    assert result.metrics['runs_successfully'] == 0
    assert 'combined_score' not in result.metrics
    assert len(transport.calls) == 4
    assert result.metrics['matrix']['succeeded_cells'] == 3
    assert 'mean' not in result.metrics['matrix']['cases'][1]
    assert result.artifacts['error_type'] == 'IncompleteMatrix'


def test_native_exception_survives_matrix_and_next_round_feedback(tmp_path):
    spec, path, _ = fixture(tmp_path)
    class FailedTransport(Transport):
        def __call__(self, task, candidate, source, output, **kwargs):
            log = output / 'retained_artifacts/algorithm/30_placement/stages/02_global_place/run.log'
            log.parent.mkdir(parents=True)
            log.write_text('startup\n' * 20000 +
                           'Traceback (most recent call last):\n'
                           '    weight_ratio = torch.nan_to_num(x)\n'
                           "AttributeError: module 'torch' has no attribute 'nan_to_num'\n")
            return EvaluationResult({'runs_successfully': 0}, {'error_message': 'Stage failed'})
    class Backend:
        contexts = []
        def ask(self, ctx):
            self.contexts.append(ctx)
            return [CandidateProposal(str(ctx.evaluations), ctx.problem.initial_program)]
        def tell(self, items):
            pass
    backend = Backend()
    result = run_task_search(backend, TaskEvaluationOptions(spec.root, tmp_path / 'search',
        open3dbench_root=SOURCE, evaluation_matrix=path), max_evaluations=2,
        evaluation_transport=FailedTransport())
    feedback = backend.contexts[1].history[0]
    assert not feedback.valid
    assert 'combined_score' not in feedback.metrics
    assert "has no attribute 'nan_to_num'" in feedback.artifacts['optimization_feedback']
    cell = feedback.metrics['matrix']['cells'][0]
    assert len(cell['native_failure_diagnostics'][0]['excerpt']) <= 1800
    assert cell['score'] is None


def test_failure_diagnostics_do_not_follow_external_symlinks(tmp_path):
    from local_bench.failure_feedback import native_failure_diagnostics
    output = tmp_path / 'cell'; output.mkdir()
    external = tmp_path / 'external.log'
    external.write_text('RuntimeError: private unrelated data\n')
    (output / 'stage.log').symlink_to(external)
    assert native_failure_diagnostics(output) == []


@pytest.mark.parametrize('retained', [False, True])
def test_failure_feedback_follows_native_pdn_log_instead_of_old_placement_warning(tmp_path, retained):
    from local_bench.failure_feedback import native_failure_diagnostics
    old = tmp_path / 'retained_artifacts/algorithm/30_placement/run.log'
    old.parent.mkdir(parents=True)
    old.write_text('[ERROR] Overlap before legalization; subsequently repaired\n')
    native = tmp_path / 'run/40_pdn/top/run.log'
    (tmp_path / 'backend-stage.log').write_text('PDNError: command failed; see ' + str(native) + '\n')
    saved = tmp_path / 'retained_artifacts/algorithm/40_pdn/top/run.log' if retained else native
    saved.parent.mkdir(parents=True)
    saved.write_text('[ERROR PDN-0001] Disconnected supply stripe\n')
    records = native_failure_diagnostics(tmp_path)
    assert len(records) == 2
    assert 'Disconnected supply stripe' in records[0]['excerpt']
    assert 'Overlap' not in str(records)


def test_missing_causal_log_does_not_blame_unrelated_stage(tmp_path):
    from local_bench.failure_feedback import native_failure_diagnostics
    old = tmp_path / 'retained_artifacts/algorithm/30_placement/run.log'
    old.parent.mkdir(parents=True)
    old.write_text('[ERROR] Old repaired overlap\n')
    (tmp_path / 'backend-stage.log').write_text('PDNError: command failed; see ' +
        str(tmp_path / 'run/40_pdn/top/run.log') + '\n')
    records = native_failure_diagnostics(tmp_path)
    assert len(records) == 1
    assert 'PDNError' in records[0]['excerpt'] and 'Old repaired' not in str(records)


def test_feedback_keeps_algorithm_activity_flags_without_boolean_scores():
    from local_bench.prompt_context import prompt_metrics
    row = prompt_metrics({'nesterov_identity_controls_only': False,
        'combined_score': True, 'placement_native_hpwl': float('nan')},
        'placement_parameters_reference_20260923')
    assert row == {'nesterov_identity_controls_only': False}
    assert prompt_metrics({'policy_diagnostics_present': True},
                          'global_routing_ripup_policy') == {'policy_diagnostics_present': True}


@pytest.mark.parametrize('current_profile', [True, False])
def test_prepared_execution_preserves_nesterov_evidence_profile(tmp_path, current_profile):
    from local_bench.evaluation import _execute_prepared
    from local_bench.current_nesterov import PROFILE
    spec = load_task(ROOT / 'local_tasks/placement_parameters_reference_20260923')
    state = dict(spec=spec, candidate={}, workspace=tmp_path / 'workspace',
        run_dir=tmp_path / 'run', resolved_config={}, open3dbench_python='python3',
        run_environment={}, artifacts={'cascade': []}, command=['native-placement'],
        command_cwd=tmp_path, output_dir=tmp_path, reference_run=None, baseline_run=None,
        timeout_seconds=30, seed_schedule=dict(io=1, partition=1002, placement=1002, routing=-1),
        current_nesterov_profile=current_profile, dreamplace_overlay=None,
        keep_workspace=True, keep_runtime=True, compact_artifacts=False, phase_binding=None)
    with mock.patch('local_bench.evaluation._run_native_flow', return_value=SimpleNamespace(returncode=0)), \
         mock.patch('local_bench.evaluation._validate_dreamplace_global_output', return_value={}), \
         mock.patch('local_bench.reference_placement_evidence.validate_reference_placement_evidence', return_value={}) as evidence, \
         mock.patch('local_bench.seeds.validate_executed_seeds', return_value={}), \
         mock.patch('local_bench.evaluation.extract_metrics', return_value={'combined_score': 1.0}):
        result = _execute_prepared(state)
    assert result.metrics['runs_successfully'] == 1
    if current_profile:
        assert evidence.call_args.kwargs['profile'] == PROFILE
        assert evidence.call_args.kwargs['placement_seed'] == 1002
    else:
        assert 'profile' not in evidence.call_args.kwargs


@pytest.mark.parametrize('change,match', [
    (lambda d: d['cells'].pop(), 'every case/seed'),
    (lambda d: d['cells'].append(d['cells'][0]), 'Duplicate'),
    (lambda d: d['seeds'].append(d['seeds'][0]), 'Duplicate seed'),
    (lambda d: d['cells'][0].update(reference_run=None), 'qualified reference'),
    (lambda d: d['cells'][0].update(seed='s2'), 'seed differs|Duplicate'),
    (lambda d: d.update(weights=[1,2]), 'requires'),
    (lambda d: d.update(task='other'), 'different local task')])
def test_bad_matrix_fails_before_any_dispatch(tmp_path, change, match):
    spec, path, data = fixture(tmp_path)
    change(data); path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        LocalEvaluationMatrix(path, spec)


def test_seed_and_input_mismatch_cannot_adopt_reference(tmp_path):
    spec, path, data = fixture(tmp_path)
    ref = Path(data['cells'][0]['reference_run']).parent / 'evaluation_result.json'
    value = json.loads(ref.read_text()); value['artifacts']['matrix_cell_contract']['seeds']['placement'] = 1002
    ref.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='mismatched matrix reference'):
        LocalEvaluationMatrix(path, spec)


@pytest.mark.parametrize('name', ['global_routing_ripup_policy', 'global_routing_maze_cost',
                                  'global_routing_iteration_controller'])
def test_routing_reference_without_hook_cannot_be_adopted(tmp_path, name):
    spec, path, data = fixture(tmp_path, name)
    ref = Path(data['cells'][0]['reference_run']).parent / 'evaluation_result.json'
    value = json.loads(ref.read_text()); value['metrics']['grt_complete'] = 0
    ref.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='required policy hook was not exercised'):
        LocalEvaluationMatrix(path, spec)


def test_unexercised_routing_reference_does_not_publish_qualified_matrix(tmp_path):
    spec, path, _ = fixture(tmp_path, 'global_routing_maze_cost', qualified=False)
    class NoHook(Transport):
        def __call__(self, *args, **kwargs):
            return EvaluationResult({'runs_successfully': 1, 'combined_score': 0,
                                     'grt_complete': 0}, {})
    result = qualify_matrix(path, spec.root, SOURCE, tmp_path / 'qualification',
                            transport=NoHook())
    assert not result.metrics['matrix']['complete']
    assert all('required policy hook was not exercised' in r['error_message']
               for r in result.metrics['matrix']['cells'])
    assert not (tmp_path / 'qualification/qualification/qualified_matrix.json').exists()


def test_ask_tell_budget_cache_and_matrix_feedback(tmp_path):
    spec, path, _ = fixture(tmp_path)
    options = TaskEvaluationOptions(spec.root, tmp_path / 'search', open3dbench_root=SOURCE,
                                    evaluation_matrix=path, matrix_workers=2)
    transport = Transport()
    class Backend:
        contexts = []; told = []
        def ask(self, ctx):
            self.contexts.append(ctx)
            return [CandidateProposal(str(ctx.evaluations), ctx.problem.initial_program)]
        def tell(self, items):
            self.told.extend(items)
    backend = Backend()
    result = run_task_search(backend, options, max_evaluations=2, evaluation_transport=transport)
    assert len(result.history) == 2 and result.history[-1].status == 'cached'
    assert len(transport.calls) == 4 and transport.preparations == 1
    assert result.history[0].metrics['matrix']['expected_cells'] == 4
    assert 'bp-ng45' in result.history[0].artifacts['optimization_feedback']
    assert 'placement_native_hpwl' in result.history[0].artifacts['optimization_feedback']
    assert '4 case/seed cells' in backend.contexts[0].problem.description
    assert backend.contexts[0].problem.metadata['cells_per_evaluation'] == 4


def test_changed_reference_cannot_hit_cached_candidate(tmp_path):
    spec, path, data = fixture(tmp_path)
    transport = Transport()
    harness = ProgramEvaluationHarness(TaskEvaluationOptions(spec.root, tmp_path / 'search',
        open3dbench_root=SOURCE, evaluation_matrix=path), evaluation_transport=transport)
    program = spec.program_template.read_text()
    assert harness.evaluate_proposal(CandidateProposal('one', program)).valid
    ref = Path(data['cells'][0]['reference_run']).parent / 'evaluation_result.json'
    ref.write_text(ref.read_text() + '\n')
    result = harness.evaluate_proposal(CandidateProposal('two', program))
    assert not result.valid and 'changed during search' in result.error_message
    assert len(transport.calls) == 4


@pytest.mark.parametrize('name', MATRIX_TASK_DIRECTORIES)
def test_seed_overrides_reach_resolved_config_and_embedded_parameters(tmp_path, name):
    spec = load_task(ROOT / 'local_tasks' / name)
    candidate = materialize_program(spec, spec.program_template.read_text(), SOURCE)
    seeds = {k:v for k,v in SEEDS[1].items() if k != 'id'}
    base_hash = sha(spec.base_config)
    result = _write_resolved_config(spec, spec.initial_candidate, tmp_path / 'config.yaml', candidate,
                                    seed_schedule=seeds)
    assert result['partition']['seed'] == result['placement']['seed'] == 1002
    for name in ('partition.json', 'placement.json'):
        file = tmp_path / 'candidate_files' / name
        if file.exists():
            row = json.loads(file.read_text())
            assert (row['partition']['seed'] if name == 'partition.json' else
                    row['concurrent_dreamplace']['initialization_seed']) == 1002
    assert sha(spec.base_config) == base_hash


def test_actual_native_seed_mismatch_is_not_silently_scored(tmp_path):
    spec = load_task(ROOT / 'local_tasks/placement_initialization')
    seeds = {k:v for k,v in SEEDS[1].items() if k != 'id'}
    file = tmp_path / '30_placement/stages/02_global_place/params.json'
    write(file, {'random_seed': 1000, 'concurrent_lol': {'initialization_seed':1000}})
    with pytest.raises(ValueError, match='Executed placement seed differs'):
        validate_executed_seeds(spec, tmp_path, seeds)


@pytest.mark.parametrize('key,value', [('io',2), ('routing',0), ('placement',True), ('partition',-1), ('placement',2**31)])
def test_bad_seed_vectors_are_rejected(key, value):
    seeds = {k:v for k,v in SEEDS[0].items() if k != 'id'}; seeds[key] = value
    with pytest.raises(ValueError): validate_schedule(seeds)


def test_no_unsafe_parallel_direct_execution(tmp_path):
    spec, path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='resource-bounded transport'):
        LocalEvaluationMatrix(path, spec).evaluate(tmp_path, SOURCE, tmp_path / 'out', workers=2)


def test_failed_qualification_never_publishes_reference_matrix(tmp_path):
    spec, path, _ = fixture(tmp_path, qualified=False)
    result = qualify_matrix(path, spec.root, SOURCE, tmp_path / 'qualification',
                            transport=Transport(fail=('bp-ng45', 1001)), workers=2)
    assert result.metrics['runs_successfully'] == 0
    assert not (tmp_path / 'qualification/qualification/qualified_matrix.json').exists()
    assert 'combined_score' not in result.metrics  # qualification is not an optimization trial


def test_qualification_passes_null_references_and_keeps_all_physical_evidence(tmp_path):
    spec, path, _ = fixture(tmp_path, qualified=False)
    transport = Transport()
    result = qualify_matrix(path, spec.root, SOURCE, tmp_path / 'qualification',
                            transport=transport, workers=2)
    assert result.metrics['runs_successfully'] == 1
    assert all(c['reference_run'] is None and not c['compact_artifacts'] for c in transport.calls)
    published = json.loads((tmp_path / 'qualification/qualification/qualified_matrix.json').read_text())
    assert len(published['cells']) == 4 and all(c['reference_run'].endswith('/result/run') for c in published['cells'])
    assert 'combined_score' not in result.metrics


def test_matrix_options_reject_single_paths_and_synthetic_modes(tmp_path):
    spec, path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='single-cell paths'):
        ProgramEvaluationHarness(TaskEvaluationOptions(spec.root, tmp_path / 'out',
            evaluation_matrix=path, baseline_run=tmp_path, open3dbench_root=SOURCE))
    with pytest.raises(ValueError, match='full mode'):
        ProgramEvaluationHarness(TaskEvaluationOptions(spec.root, tmp_path / 'out',
            evaluation_matrix=path, evaluation_mode='verify', open3dbench_root=SOURCE))


def test_dated_nesterov_profile_cannot_change_seed(tmp_path):
    from local_bench.reference_placement_evidence import validate_reference_placement_evidence, DYNAMIC_TASK
    # Its parameter validator stays fixed; matrix current-source profile is explicit.
    from local_bench.current_nesterov import contract
    recipe = ROOT / 'local_bench/reference_configs/nesterov_current.yaml'
    cfg = yaml.safe_load(recipe.read_text())
    cfg['placement']['seed'] = 1002; cfg['partition']['seed'] = 1002
    path = tmp_path / 'current.yaml'; path.write_text(yaml.safe_dump(cfg))
    with mock.patch.dict('os.environ', {'OPEN3DBENCH_IMAGE_ID':'sha256:'+'a'*64}):
        with pytest.raises(ValueError, match='recipe differs'):
            contract(path)
        result = contract(path, {k:v for k,v in SEEDS[1].items() if k != 'id'})
        assert result['config']['placement']['seed'] == 1002


def test_placement_matrix_command_uses_supported_native_arguments(tmp_path):
    import importlib.util
    import sys
    from local_bench.evaluation import _dreamplace_global_placement_command
    spec = load_task(ROOT / 'local_tasks/placement_initialization')
    candidate = materialize_program(spec, spec.program_template.read_text(), SOURCE)
    config = _write_resolved_config(spec, spec.initial_candidate, tmp_path/'config.yaml', candidate,
                                   seed_schedule={k:v for k,v in SEEDS[1].items() if k != 'id'})
    for name in ('20_partition/partition_assignment.csv','20_partition/die_regions.json',
                 '10_floorplan/design_params.json'):
        p=tmp_path/'run'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    command, _ = _dreamplace_global_placement_command(spec, config, SOURCE, tmp_path/'run', 'python3')
    sys.path.insert(0,str(SOURCE/'Place-3D'))
    try:
        from src.placement.concurrent_dreamplace import build_parser
        args = build_parser().parse_args(command[3:])
        data = json.loads(Path(args.flow_config).read_text())
        assert data['concurrent_dreamplace']['initialization_seed'] == 1002
    finally:
        sys.path.pop(0)


def test_native_prefix_seed_evidence_is_hash_bound(tmp_path):
    from local_bench.seeds import validate_prefix
    spec=load_task(ROOT/'local_tasks/placement_initialization')
    cfg={'design':'bp_fe','type':'lol','pdk':'ng45'}
    seeds={k:v for k,v in SEEDS[0].items() if k!='id'}
    files={'floorplan':('io_metrics','io_metrics.json',{'seed':1}),
           'partition':('method_manifest','partition_manifest.json',{'seed':1001})}
    for stage in spec.upstream_stages:
        directory=tmp_path/UPSTREAM_STAGE_DIRS[stage];directory.mkdir()
        artifacts={}
        if stage in files:
            key,name,data=files[stage];p=directory/name;write(p,data)
            artifacts[key]={'path':name,'bytes':p.stat().st_size,'sha256':sha(p)}
        write(directory/'stage_manifest.json',dict(status='success',design='bp_fe',integration='lol',pdk='ng45',artifacts=artifacts))
    assert len(validate_prefix(spec,tmp_path,cfg,seeds))==3
    with pytest.raises(ValueError,match='Frozen partition seed differs'):
        validate_prefix(spec,tmp_path,cfg,{**seeds,'partition':1002})
    path=tmp_path/'20_partition/partition_manifest.json'
    path.write_text('{"seed":1002}')
    with pytest.raises(ValueError,match='Seed evidence changed'):
        validate_prefix(spec,tmp_path,cfg,seeds)


def test_nonfinite_cell_invalidates_complete_candidate(tmp_path):
    spec,path,_=fixture(tmp_path)
    def execute(*args,**kwargs):
        return EvaluationResult({'runs_successfully':1,'combined_score':float('nan')},{})
    result=LocalEvaluationMatrix(path,spec).evaluate(tmp_path,SOURCE,tmp_path/'out',transport=execute)
    assert 'combined_score' not in result.metrics and result.metrics['runs_successfully']==0
    assert all(row['status']=='failed' for row in result.metrics['matrix']['cells'])


def test_cli_refuses_to_overwrite_an_existing_matrix_candidate(tmp_path):
    from local_bench.cli import main
    spec,path,_=fixture(tmp_path)
    output=tmp_path/'existing';output.mkdir()
    candidate=output/'input_candidate/candidate.json';candidate.parent.mkdir()
    candidate.write_text('preserved original evidence')
    with pytest.raises(FileExistsError,match='new empty output'):
        main(['evaluate',str(spec.root),'--matrix',str(path),'--open3dbench-root',str(SOURCE),
              '--output-dir',str(output)])
    assert candidate.read_text()=='preserved original evidence'


@pytest.mark.parametrize('name', ['tp_gnn_binary_loss_timing', 'tp_gnn_loss_decoder_timing'])
def test_new_single_seed_gnn_tasks_do_not_inherit_historical_matrix_authority(tmp_path, name):
    spec, path, _ = fixture(tmp_path, name)
    with pytest.raises(ValueError, match='no defined seed-matrix contract'):
        LocalEvaluationMatrix(path, spec)


def test_uncertain_cell_is_retained_and_next_cell_is_not_dispatched(tmp_path):
    from grt_bench.runtime import UncertainExecution
    spec, path, _ = fixture(tmp_path)
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(kwargs)
        raise UncertainExecution("Docker termination is unconfirmed")
    with pytest.raises(UncertainExecution):
        LocalEvaluationMatrix(path, spec).evaluate(tmp_path, SOURCE, tmp_path / "out",
            transport=interrupted, workers=1, timeout_seconds=None)
    assert len(calls) == 1 and calls[0]["timeout_seconds"] is None
    cells = list((tmp_path / "out/cells").glob("*/*/cell.json"))
    assert len(cells) == 1
    row = json.loads(cells[0].read_text())
    assert row["status"] == "uncertain" and "termination" in row["error_message"]
