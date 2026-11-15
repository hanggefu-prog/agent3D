"""Trusted fixed-cell clock ECO executor. No candidate Tcl is accepted."""
import json
import os
from pathlib import Path
import subprocess
import uuid

from engineering_bench.interfaces import validate_clock_plan
from engineering_bench.native_snapshot import bind, platform_libs
from engineering_bench.protocol import sha, write_new, now


def quote(value):
    value = str(value)
    if any(c in value for c in '\r\n\0'):
        raise ValueError('Multiline Tcl identifier')
    return '"' + ''.join('\\' + c if c in '\\"$[]' else c for c in value) + '"'


PREFIX = r'''
set_thread_count 8
foreach lib [split $::env(ENG_LIBS) "\n"] { read_liberty $lib }
read_db $::env(ENG_ODB)
set db [ord::get_db]
set block [ord::get_db_block]
set originals [dict create]
foreach inst [$block getInsts] {
  dict set originals [$inst getName] [list [$inst getLocation] [$inst getOrient] [$inst getPlacementStatus] [[$inst getMaster] getName]]
  $inst setPlacementStatus FIRM
}
proc insert_chain {target master sites tag} {
  global db block
  set cut [string last / $target]
  set inst [$block findInst [string range $target 0 [expr {$cut-1}]]]
  if {$inst eq "NULL"} {error "missing clock branch"}
  set pin [$inst findITerm [string range $target [expr {$cut+1}] end]]
  if {$pin eq "NULL" || [$pin getIoType] ne "INPUT"} {error "invalid branch input"}
  set net [$pin getNet]
  if {$net eq "NULL"} {error "floating branch input"}
  set lib [$db findMaster $master]
  if {$lib eq "NULL"} {error "missing buffer master"}
  $pin disconnect
  set index 0
  foreach site $sites {
    set name ${tag}_${index}
    if {[$block findInst $name] ne "NULL" || [$block findNet ${name}_net] ne "NULL"} {error "ECO name collision"}
    set added [odb::dbInst_create $block $lib $name]
    if {$added eq "NULL"} {error "buffer creation failed"}
    odb::dbIntProperty_create $added OPENROAD_CTS_BUFFER 1
    $added setLocation {*}$site
    $added setPlacementStatus PLACED
    set input [$added findITerm A]
    set output [$added findITerm Z]
    if {$input eq "NULL" || $output eq "NULL"} {error "unexpected buffer interface"}
    $input connect $net
    set net [odb::dbNet_create $block ${name}_net]
    $net setSigType CLOCK
    $output connect $net
    foreach pg [$added getITerms] {
      set type [[$pg getMTerm] getSigType]
      if {$type in {POWER GROUND}} {
        set expected [expr {$type eq "POWER" ? "VDD" : "VSS"}]
        set rail [$block findNet $expected]
        if {$rail eq "NULL"} {error "missing PG rail"}
        $pg connect $rail
      }
    }
    incr index
  }
  $pin connect $net
}
'''

SUFFIX = r'''
detailed_placement -max_displacement {500 500}
check_placement -verbose
dict for {name old} $originals {
  set inst [$block findInst $name]
  lassign $old xy orient status master
  if {[$inst getLocation] ne $xy || [$inst getOrient] ne $orient || [[$inst getMaster] getName] ne $master} {error "original cell moved: $name"}
  $inst setPlacementStatus $status
}
set pg_count 0
foreach inst [$block getInsts] {
  foreach pin [$inst getITerms] {
    set type [[$pin getMTerm] getSigType]
    if {$type ni {POWER GROUND}} {continue}
    set rail [$pin getNet]
    set expected [expr {$type eq "POWER" ? "VDD" : "VSS"}]
    if {$rail eq "NULL" || [$rail getName] ne $expected} {error "invalid PG"}
    incr pg_count
  }
}
write_db /out/design.odb
puts "ENGINEERING_CLOCK_ECO_OK originals=[dict size $originals] pg=$pg_count"
'''


def execute(*, context, plan, reference, platform, image, output):
    verified = validate_clock_plan(context, plan)
    output.mkdir(parents=True, exist_ok=False)
    additions = []
    for die in ('bottom', 'top'):
        folder = output / die
        folder.mkdir()
        script = PREFIX
        for index, action_id in enumerate(verified['actions']):
            action = context['actions'][action_id]
            if action['die'] != die:
                continue
            if action['master'] not in ('BUF_X1_' + die, 'BUF_X4_' + die):
                raise ValueError('Unsupported physical action master')
            sites = action['sites']
            if len(sites) != action['added_buffers'] or any(type(s[k]) is not int for s in sites for k in ('x_dbu', 'y_dbu')):
                raise ValueError('Malformed action geometry')
            tag = 'engineering_clock_' + str(index)
            coords = '[list ' + ' '.join('[list %d %d]' % (s['x_dbu'], s['y_dbu']) for s in sites) + ']'
            script += '\ninsert_chain %s %s %s %s\n' % (quote(action['target_input_pin']), quote(action['master']), coords, quote(tag))
            additions += [(die, tag + '_' + str(i)) for i in range(len(sites))]
        (folder / 'execute.tcl').write_text(script + SUFFIX)
        odb = reference / ('normalized-' + die) / 'native/design.odb'
        name = 'o3db-engineering-clock-eco-' + uuid.uuid4().hex[:12]
        cmd = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--cpus', '8', '--memory', '32g', '--pids-limit', '2048',
            '--tmpfs', '/tmp:rw,size=2g', '--user', str(os.getuid()) + ':' + str(os.getgid())]
        for host, target, ro in [(odb, '/input.odb', True), (platform, '/platform', True), (folder, '/out', False)]:
            cmd += bind(host, target, ro)
        cmd += ['--env', 'HOME=/tmp', '--env', 'ENG_ODB=/input.odb', '--env',
            'ENG_LIBS=' + '\n'.join('/platform/' + str(p.relative_to(platform)) for p in platform_libs(platform, die)),
            '--entrypoint', '/usr/local/bin/openroad', image, '-no_init', '-exit', '/out/execute.tcl']
        write_new(folder / 'dispatch.json', {'command': cmd, 'input_sha256': sha(odb), 'script_sha256': sha(folder / 'execute.tcl'), 'started_at': now()})
        with (folder / 'native.log').open('x') as log:
            try:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, timeout=20)
                raise
        if code or 'ENGINEERING_CLOCK_ECO_OK' not in (folder / 'native.log').read_text():
            raise ValueError('Native clock action/legality failed: ' + die)
        write_new(folder / 'result.json', {'status': 'succeeded', 'odb_sha256': sha(folder / 'design.odb'),
            'log_sha256': sha(folder / 'native.log'), 'completed_at': now()})
    write_new(output / 'result.json', {'status': 'succeeded', 'additions': additions, 'validated_plan': verified})
    return additions


def sample_plan(subject, context_path, image, output):
    """Execute arbitrary submitted Python only in an offline, read-only sandbox."""
    output.mkdir(parents=True, exist_ok=False)
    code = ('import json,sys; sys.path.insert(0,"/subject"); '
        'from engineering.clock_compensation import plan_clock_compensation; '
        'context=json.load(open("/context.json")); plan=plan_clock_compensation(context); '
        'json.dump(plan,open("/out/plan.json","x"))')
    name = 'o3db-engineering-clock-policy-' + __import__('uuid').uuid4().hex[:12]
    cmd = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--cpus', '2', '--memory', '8g', '--pids-limit', '128',
        '--tmpfs', '/tmp:rw,size=256m', '--user', str(os.getuid()) + ':' + str(os.getgid())]
    for host, target, ro in [(subject, '/subject', True), (context_path, '/context.json', True), (output, '/out', False)]:
        cmd += bind(host, target, ro)
    cmd += ['--env', 'PYTHONDONTWRITEBYTECODE=1', '--entrypoint', 'python3', image, '-c', code]
    write_new(output / 'dispatch.json', {'command': cmd, 'context_sha256': sha(context_path), 'started_at': now()})
    with (output / 'policy.log').open('x') as log:
        try:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        except BaseException:
            from grt_bench.runtime import stop_and_confirm
            stop_and_confirm(name, output)
            raise
    if p.returncode:
        raise ValueError('Clock policy execution failed')
    return json.loads((output / 'plan.json').read_text())
