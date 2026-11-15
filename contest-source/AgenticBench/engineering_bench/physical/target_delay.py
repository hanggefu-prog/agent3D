"""Protected fixed-slew driver-through-original-sink graph-delay measurement.

This is a characterization view, separate from unmodified global setup/hold STA.
No delay arc is assigned. Only the frozen original driver input slew is fixed,
identically for reference and candidate; all cell/wire/HBT delays remain native.
"""
import csv
import json
import math
from pathlib import Path


TARGET_DELAY_TCL = r'''
set scene [lindex [sta::get_scenes *] 0]
set requests [open $::env(ENG_TARGETS) r]
set records {}
while {[gets $requests line] >= 0} {
  if {$line eq ""} { continue }
  lassign [split $line "\t"] target driver_input driver_output sink allowed
  set pins [get_pins -quiet [list $driver_input]]
  if {[llength $pins] != 1} { error "missing fixed driver input $driver_input" }
  set_assigned_transition $::env(ENG_FIXED_SLEW_NS) $pins
  lappend records [list $target $driver_input $driver_output $sink $allowed]
}
close $requests
sta::find_timing
set result [open $::env(ENG_DELAYS) w]
puts $result "target\tsink\trise_delay_ns\tfall_delay_ns\tworst_delay_ns"

foreach row $records {
  lassign $row target driver_input driver_output sink allowed
  set allowed_cells [split $allowed ,]
  set start [lindex [[get_pins [list $driver_input]] vertices] 0]
  set delays [dict create]
  foreach input_edge {rise fall} {
    set queue [list [list $start $input_edge 0.0 0]]
    set visited [dict create]
    set limit 0
    while {[llength $queue]} {
      lassign [lindex $queue 0] vertex transition arrival passed_driver
      set queue [lrange $queue 1 end]
      incr limit
      if {$limit > 20000} { error "unsupported fanout graph size/cycle" }
      set name [get_full_name [$vertex pin]]
      set key "$name/$transition/$passed_driver"
      if {[dict exists $visited $key] && [dict get $visited $key] >= $arrival} { continue }
      dict set visited $key $arrival
      if {$name eq $sink && $passed_driver} {
        if {![dict exists $delays $input_edge] || [dict get $delays $input_edge] < $arrival} {
          dict set delays $input_edge $arrival
        }
        continue
      }
      set iter [$vertex out_edge_iterator]
      while {[$iter has_next]} {
        set arc_edge [$iter next]
        if {[$arc_edge is_disabled_loop] || [$arc_edge is_disabled_constraint] || [$arc_edge is_disabled_constant]} { continue }
        set dest [$arc_edge to]
        set dest_name [get_full_name [$dest pin]]
        set role [$arc_edge role]
        if {$role ne "wire"} {
          if {$role ne "combinational" || [get_full_name [[$vertex pin] instance]] ni $allowed_cells} { continue }
        }
        foreach arc [$arc_edge timing_arcs] {
          if {[$arc from_edge_name] ne $transition} { continue }
          set delay [sta::time_sta_ui [$arc_edge arc_delay $arc $scene max]]
          if {![string is double -strict $delay] || abs($delay) > 1e6} { error "nonfinite native arc delay" }
          set passed [expr {$passed_driver || $dest_name eq $driver_output}]
          lappend queue [list $dest [$arc to_edge_name] [expr {$arrival+$delay}] $passed]
        }
      }
      $iter finish
    }
  }
  if {![dict exists $delays rise] || ![dict exists $delays fall]} { error "target $target lacks a physical driver-through-sink path to $sink" }
  set rise [dict get $delays rise]
  set fall [dict get $delays fall]
  puts $result [join [list $target $sink $rise $fall [expr {max($rise,$fall)}]] "\t"]
}
close $result
'''


def measured_receivers(target):
    """Remote receivers, plus source-die local receivers when the context lists them.

    Contexts created before T3 v2 have no original_local_receivers and keep
    the frozen remote-only request set; the Tcl query needs no change because
    local receivers hang off the same original driver output.
    """
    return sorted(set(target['original_remote_receivers']) | set(target.get('original_local_receivers', ())))


def requests(context, added_cells=None):
    rows = []
    added_cells = added_cells or {}
    for name, target in sorted(context['target_nets'].items()):
        driver = target['original_driver']
        inputs = context['driver_input_pins'][driver]
        if len(inputs) != 1:
            raise ValueError('Initial target contract requires one-input original drivers')
        allowed = [driver.rsplit('/', 1)[0], *added_cells.get(name, [])]
        for sink in measured_receivers(target):
            row = [name, inputs[0], driver, sink, ','.join(allowed)]
            if any(any(c in v for c in ('\t', '\n', '\r')) for v in row):
                raise ValueError('Unsupported TSV pin identity')
            rows.append('\t'.join(row))
    return '\n'.join(rows) + '\n'


def parse_delays(path, context):
    result = {name: {} for name in context['target_nets']}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            name, sink = row['target'], row['sink']
            if name not in result or sink not in measured_receivers(context['target_nets'][name]) or sink in result[name]:
                raise ValueError('Unexpected/repeated target receiver')
            values = [float(row[k]) for k in ('rise_delay_ns', 'fall_delay_ns', 'worst_delay_ns')]
            if not all(math.isfinite(v) and v >= 0 for v in values) or abs(max(values[:2])-values[2]) > 1e-9:
                raise ValueError('Invalid native receiver delay')
            result[name][sink] = values[2]
    if any(set(result[n]) != set(measured_receivers(t)) for n, t in context['target_nets'].items()):
        raise ValueError('Missing fixed receiver measurement')
    return result
