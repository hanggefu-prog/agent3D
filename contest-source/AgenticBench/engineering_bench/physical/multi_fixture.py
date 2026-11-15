"""Independent deterministic one-extra-HBT native fixture; never scored/search."""
import json
import os
from pathlib import Path
import subprocess
import uuid

from clock_executor import PREFIX, SUFFIX, quote
from engineering_bench.native_snapshot import bind, platform_libs
from engineering_bench.protocol import write_new, sha


NEW_ANCHOR = r'''
set anchor [odb::dbInst_create $block [$db findMaster $master] $hbt]
if {$anchor eq "NULL"} {error "HBT creation failed"}
$anchor setLocation $x $y
$anchor setPlacementStatus FIRM
set term [odb::dbBTerm_create $net $port]
$term setIoType $direction
$term setSigType SIGNAL
set layer [[ord::get_db_tech] findLayer $layer_name]
set width [$layer getWidth]
set box [odb::dbBPin_create $term]
$box setPlacementStatus FIRM
set lx [expr {$x+($hw-$width)/2}]
set ly [expr {$y+($hh-$width)/2}]
odb::dbBox_create $box $layer $lx $ly [expr {$lx+$width}] [expr {$ly+$width}]
'''


def fixture(context, *, reference, platform, image, output):
    output.mkdir(parents=True, exist_ok=False)
    selected = sorted(context['target_nets'])[0]
    target = context['target_nets'][selected]
    site_id = context['spare_site_ids'][0]
    site = context['site_geometry'][site_id]
    direction = 'B2T' if target['source_die'] == 'bottom' else 'T2B'
    hbt = 'HBT_' + direction + '_' + str(context['new_hbt_id_base'])
    buffer = 'engineering_fixture_buffer'
    sd, td = target['source_die'], target['sink_die']
    moved_sink = target['original_remote_receivers'][-1]
    native = context['native']
    plan = {'target_nets': [], 'branch_map': {}}
    for index, (name, t) in enumerate(sorted(context['target_nets'].items())):
        receivers = t['original_remote_receivers']
        branch_id = 'fixture_' + str(index)
        branches = [{'branch_id': branch_id, 'source_die': t['source_die'], 'sink_die': t['sink_die'],
            'hbt_id': t['original_hbt_id'], 'site_id': t['original_site_id'], 'buffer_instance': None}]
        if name == selected:
            branches.append({'branch_id': branch_id + '_extra', 'source_die': sd, 'sink_die': td,
                'hbt_id': hbt, 'site_id': site_id, 'buffer_instance': buffer})
        plan['target_nets'].append({'logical_net': name, 'branches': branches})
        plan['branch_map'][name] = [b['branch_id'] for b in branches]
    write_new(output / 'branch_plan.json', plan)
    for die in ('bottom', 'top'):
        folder = output / die
        folder.mkdir()
        script = PREFIX
        values = {'hbt': hbt, 'x': site['x_dbu'], 'y': site['y_dbu'], 'hw': native['hbt_width_dbu'],
            'hh': native['hbt_height_dbu'], 'master': native[die + '_hbt_master'], 'layer_name': native[die + '_hbt_layer'],
            'port': hbt + ('_SRC' if die == sd else '_DST'), 'direction': 'OUTPUT' if die == sd else 'INPUT'}
        script += '\n' + '\n'.join('set %s %s' % (k, quote(v)) for k, v in values.items()) + '\n'
        script += 'set net [odb::dbNet_create $block engineering_fixture_net]\n'
        if die == sd:
            driver = native['driver_pins'][target['original_driver']]['iterm']
            pos = site['buffer_by_die'][die]
            script += 'set driver [$block findITerm ' + quote(driver) + ']\n'
            script += 'set buffer [odb::dbInst_create $block [$db findMaster ' + quote('BUF_X1_' + die) + '] ' + quote(buffer) + ']\n'
            script += '$buffer setLocation %d %d\n' % (pos['x_dbu'], pos['y_dbu'])
            script += '''$buffer setPlacementStatus PLACED
[$buffer findITerm A] connect [$driver getNet]
[$buffer findITerm Z] connect $net
[$buffer findITerm VDD] connect [$block findNet VDD]
[$buffer findITerm VSS] connect [$block findNet VSS]
'''
        else:
            pin = native['receiver_pins'][moved_sink]['iterm']
            script += '[$block findITerm ' + quote(pin) + '] connect $net\n'
        script += NEW_ANCHOR + SUFFIX
        (folder / 'fixture.tcl').write_text(script)
        name = 'o3db-engineering-hbt-fixture-' + uuid.uuid4().hex[:12]
        cmd = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--cpus', '8', '--memory', '32g', '--pids-limit', '2048',
            '--tmpfs', '/tmp:rw,size=2g', '--user', str(os.getuid()) + ':' + str(os.getgid())]
        for host, dest, ro in [(reference / ('normalized-' + die) / 'native/design.odb', '/input.odb', True),
            (platform, '/platform', True), (folder, '/out', False)]:
            cmd += bind(host, dest, ro)
        cmd += ['--env', 'ENG_ODB=/input.odb', '--env', 'HOME=/tmp', '--env',
            'ENG_LIBS=' + '\n'.join('/platform/' + str(p.relative_to(platform)) for p in platform_libs(platform, die)),
            '--entrypoint', '/usr/local/bin/openroad', image, '-no_init', '-exit', '/out/fixture.tcl']
        write_new(folder / 'dispatch.json', {'command': cmd, 'script_sha256': sha(folder / 'fixture.tcl')})
        with (folder / 'native.log').open('x') as log:
            try:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, timeout=20)
                raise
        if code or 'ENGINEERING_CLOCK_ECO_OK' not in (folder / 'native.log').read_text():
            raise ValueError('Independent HBT materialization fixture failed: ' + die)
    return plan
