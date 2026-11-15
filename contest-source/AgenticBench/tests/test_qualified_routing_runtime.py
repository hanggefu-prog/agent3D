"""Transport regressions; mocks are not physical qualification."""
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.qualified_routing_runtime import QualifiedRoutingRuntime
from scripts.qualified_local_runtime import QualifiedLocalRuntime
from scripts.run_local_task_smoke import LocalSmokeRuntime


@pytest.fixture
def runtime(tmp_path):
    r = object.__new__(QualifiedRoutingRuntime)
    r.batch = tmp_path/'runs'
    r.source = tmp_path/'source'
    r.framework = tmp_path/'framework'
    r.write_root = r.batch/'job'
    r.equivalence_proof = tmp_path/'release'/'runtime-audit.json'
    r.equivalence_proof_sha256 = 'a'*64
    r.image_preparation = lambda gpu: None
    return r


def test_candidate_image_and_limits_unchanged(runtime):
    base = ['docker','run','--network','none','--cpus','16','--memory','48g',
            '--gpus','device=1','-v','/reference:/reference:ro','sha256:candidate','evaluate']
    with patch.object(runtime,'_validate_address') as validate, patch.object(
            LocalSmokeRuntime,'docker',return_value=base.copy()) as docker:
        result = runtime.docker(['evaluate'],1)
    assert result[:2]+result[4:] == base
    assert result[2:4] == ['--mount',f'type=bind,src={runtime.equivalence_proof},dst={runtime.equivalence_proof},readonly']
    assert validate.call_count == 2
    docker.assert_called_once_with(runtime,['evaluate'],1,None)


@pytest.mark.parametrize('target',['batch','source','framework','write_root'])
def test_overlap_rejected_before_build(runtime,target):
    runtime.equivalence_proof = getattr(runtime,target)/'proof.json'
    with patch.object(runtime,'_validate_address'), patch.object(LocalSmokeRuntime,'docker') as docker:
        with pytest.raises(ValueError,match='overlaps'): runtime.docker([],1)
        docker.assert_not_called()


@pytest.mark.parametrize('bad',['a,b','a:b','a\nb','a\rb'])
def test_unsafe_mount_paths(runtime,bad):
    runtime.equivalence_proof = runtime.equivalence_proof.parent/bad
    with patch.object(runtime,'_validate_address'), pytest.raises(ValueError,match='Unsafe'):
        runtime.docker([],1)


def test_invalid_proof_fails_before_build(runtime):
    with patch.object(runtime,'_validate_address',side_effect=ValueError('proof changed')), patch.object(LocalSmokeRuntime,'docker') as docker:
        with pytest.raises(ValueError,match='proof changed'): runtime.docker([],1)
        docker.assert_not_called()


def test_base_image_path_unchanged(runtime):
    runtime.image_preparation = None
    with patch.object(runtime,'_validate_address'), patch.object(QualifiedLocalRuntime,'docker',return_value=['base']) as base:
        assert runtime.docker([],1) == ['base']
        base.assert_called_once_with([],1,None)


def test_proof_changed_during_build_rejected(runtime):
    with patch.object(runtime,'_validate_address',side_effect=[None,ValueError('changed after build')]), patch.object(
            LocalSmokeRuntime,'docker',return_value=['docker','run','candidate','evaluate']):
        with pytest.raises(ValueError,match='changed after build'): runtime.docker([],1)


def test_scratch_overlap_rejected(runtime):
    with patch.object(runtime,'_validate_address'), pytest.raises(ValueError,match='overlaps'):
        runtime.docker([],1,runtime.equivalence_proof.parent)
