"""Exercise the actual Docker command builder, without starting Docker."""
from types import SimpleNamespace
import json

import multi_final_queue as runtime


def test_private_scratch_does_not_weaken_readonly_inputs(tmp_path, monkeypatch):
    subject, reference = tmp_path / 'subject', tmp_path / 'reference'
    subject.mkdir()
    context = tmp_path / 'context.json'
    context.write_text('{}')
    for die in ('bottom', 'top'):
        path = reference / ('normalized-' + die) / 'native/design.odb'
        path.parent.mkdir(parents=True)
        path.write_text('fixture')
    output = tmp_path / 'output'
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        (output / 'eco').mkdir()
        (output / 'eco/branch_plan.json').write_text(json.dumps({'fixture': True}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runtime.subprocess, 'run', fake_run)
    assert runtime.synthesize(subject, context, reference, 'sha256:' + 'a' * 64, output) == {'fixture': True}
    assert (output / 'scratch').is_dir()
    command = commands[0]
    assert '--read-only' in command
    assert command[command.index('--network') + 1] == 'none'
    mounts = [command[i+1] for i, arg in enumerate(command[:-1]) if arg == '--mount']
    assert any('dst=/scratch,readonly=false' in m and str(output / 'scratch') in m for m in mounts)
    assert any('dst=/subject,readonly=true' in m for m in mounts)
    assert any('dst=/context.json,readonly=true' in m for m in mounts)
