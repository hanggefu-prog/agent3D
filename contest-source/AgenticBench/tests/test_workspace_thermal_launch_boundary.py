"""Host orchestration must not import physical code before Docker starts."""
import builtins
from pathlib import Path
import pytest
from scripts.workspace_codesign_evaluator import WorkspaceEvaluator
from local_bench import reference_catalog
from scripts import workspace_latest_reference as latest

def test_latest_thermal_command_without_host_physical_package(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='open3dbench' or name.startswith('open3dbench.'):
            raise ModuleNotFoundError(name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    obj=object.__new__(WorkspaceEvaluator)
    obj.latest=True
    command=obj._thermal_command(Path('/trusted/source'),Path('/candidate/run'),{'pdk':'ng45'})
    assert command==['/usr/bin/python3','-m','scripts.workspace_cts_thermal',
        '--checkpoint','/candidate/run/70_cts/checkpoint.json',
        '--platform-dir','/trusted/source/OpenROAD-3D/flow/platforms/nangate45_3D',
        '--hotspot-root','/opt/open3dbench-assets/hotspot',
        '--output-dir','/candidate/run/95_thermal_snapshot']

@pytest.mark.parametrize('kind',[latest.KIND,latest.THERMAL_KIND])
def test_retention_dispatches_only_exact_latest_producer(monkeypatch,kind):
    producer=dict(kind=kind,case='bp',benchmark_input='/input',run_root='/reference',task_id='task')
    if kind==latest.THERMAL_KIND: producer['thermal_root']='/thermal'
    observed=[]
    def verify(value):
        observed.append(value)
        raise ValueError('independent validator rejected changed physical evidence')
    monkeypatch.setattr(latest,'qualify_producer',verify)
    with pytest.raises(ValueError,match='independent validator'):
        reference_catalog.qualify_producer(producer)
    assert observed==[producer]
    with pytest.raises(ValueError,match='unknown latest'):
        reference_catalog.qualify_producer({**producer,'bypass':True})
