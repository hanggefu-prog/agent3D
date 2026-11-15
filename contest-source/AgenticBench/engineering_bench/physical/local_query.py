"""Protected native queries against a private copy of an existing STA view."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

EXCEPTIONS = r'''
set result [open /out/excluded-pins.tsv w]
puts $result "pin\treason"
sta::find_timing
# A logically constant net has no active propagation path. Exclude it before
# ranking/freezing targets, not after a reference or candidate measurement.
foreach pin [get_pins -hierarchical *] {
  set value [sta::pin_sim_logic_value $pin]
  if {$value eq "0" || $value eq "1"} {
    puts $result "[get_full_name $pin]\tconstant-propagation-$value"
  }
}
set latch [all_registers -data_pins -level_sensitive]
if {[llength $latch]} {
  foreach pin [get_fanin -flat -trace_arcs all -to $latch] {
    puts $result "[get_full_name $pin]\tlatch-max-min-delay-cone"
  }
}
set reset [get_ports -quiet rst_ni]
if {[llength $reset]} {
  foreach pin [get_fanout -flat -trace_arcs all -from $reset] {
    puts $result "[get_full_name $pin]\tfalse-path-reset-cone"
  }
}
set jtag [get_clocks -quiet jtag_clock]
if {[llength $jtag]} {
  set sinks [all_registers -clock $jtag -data_pins]
  set starts [all_registers -clock $jtag -output_pins]
  set ports [get_ports -quiet jtag_tdo]
  if {[llength $ports]} {set sinks [concat $sinks $ports]}
  if {[llength $sinks]} {
    foreach pin [get_fanin -flat -trace_arcs all -to $sinks] {
      puts $result "[get_full_name $pin]\tasynchronous-jtag-cone"
    }
  }
  if {[llength $starts]} {
    foreach pin [get_fanout -flat -trace_arcs all -from $starts] {
      puts $result "[get_full_name $pin]\tasynchronous-jtag-cone"
    }
  }
}
close $result
'''


def exception_query_digest():
    import hashlib
    return hashlib.sha256(EXCEPTIONS.encode()).hexdigest()


def reusable_exception(receipt, evidence):
    from engineering_bench.protocol import sha
    return (receipt.get('status') == 'succeeded'
        and receipt.get('exception_query_sha256') == exception_query_digest()
        and receipt.get('result_sha256') == sha(evidence))


def query(*, source, platform, run, measurement_root, image, output, mode, context=None, added_cells=None):
    from measure_host import bind
    from engineering_bench.protocol import write_new, sha, now
    from cts_final_queue import check_artifacts
    old = measurement_root / 'native/measurement'
    check_artifacts(old, json.loads((old / 'result.json').read_text()))
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'native'
    shutil.copytree(measurement_root / 'native', work)
    if context:
        from target_delay import requests
        (work / 'requests.tsv').write_text(requests(context, added_cells))
    name = 'o3db-engineering-query-' + uuid.uuid4().hex[:12]
    cmd = ['docker', 'run', '--rm', '--name', name, '--read-only', '--network', 'none', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--cpus', '8', '--memory', '32g', '--pids-limit', '2048',
        '--tmpfs', '/tmp:rw,size=2g', '--user', str(os.getuid()) + ':' + str(os.getgid())]
    for host, target, ro in [(source, '/physical', True), (platform, '/platform', True),
        (run, '/input', True), (Path(__file__).parent, '/adapter', True), (work, '/out', False)]:
        cmd += bind(host, target, ro)
    cmd += ['--env', 'HOME=/tmp', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONPATH=/physical',
        '--env', 'OPEN3DBENCH_THREADS=8', '--entrypoint', 'python3', image,
        '/adapter/local_query.py', '--native', mode]
    if context:
        cmd += ['--slew', str(context['reference_input_slew_ns'])]
    write_new(output / 'dispatch.json', {'command': cmd, 'input_measurement_sha256': sha(old / 'result.json'), 'started_at': now()})
    with (output / 'native.log').open('x') as log:
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, timeout=20)
            raise
    if code:
        raise ValueError('Native local query failed: ' + str(output))
    result = work / ('excluded-pins.tsv' if mode == 'exceptions' else 'target-delays.tsv')
    write_new(output / 'result.json', {'status': 'succeeded', 'mode': mode, 'result_sha256': sha(result),
        'exception_query_sha256': exception_query_digest() if mode == 'exceptions' else None, 'completed_at': now()})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native', choices=('exceptions', 'delays'), required=True)
    p.add_argument('--slew', type=float)
    a = p.parse_args()
    sys.path.insert(0, '/physical')
    from open3dbench.backend.three_d_opt import _run_openroad
    from target_delay import TARGET_DELAY_TCL
    script = Path('/out/measurement/assembly/passive/validate_sta.tcl').read_text()
    suffix = EXCEPTIONS if a.native == 'exceptions' else TARGET_DELAY_TCL
    _run_openroad(script + '\n' + suffix, {'ENG_TARGETS': '/out/requests.tsv', 'ENG_FIXED_SLEW_NS': str(a.slew),
        'ENG_DELAYS': '/out/target-delays.tsv'}, Path('/out/query.log'), '/usr/local/bin/openroad')
