"""Read-only physical geometry and current joint-STA features for local tasks."""
import argparse
import csv
import json
from pathlib import Path
import sys


GEOMETRY = r'''
read_db $::env(ENG_ODB)
foreach lib [split $::env(ENG_LIBS) "\n"] { if {$lib ne ""} { read_liberty $lib } }
set block [ord::get_db_block]
set out [open $::env(ENG_GEOMETRY) w]
puts $out "kind\tname\tx\ty\tw\th\textra"
puts $out [join [list units dbu [[$block getDb] getTech] 0 0 0 [$block getDbUnitsPerMicron]] "\t"]
foreach inst [$block getInsts] {
  set box [$inst getBBox]
  set kind [expr {[[$inst getMaster] getType] eq "COVER" ? "cover" : "cell"}]
  puts $out [join [list $kind [$inst getName] [$box xMin] [$box yMin] [expr {[$box xMax]-[$box xMin]}] [expr {[$box yMax]-[$box yMin]}] [[$inst getMaster] getName]] "\t"]
  foreach pin [$inst getITerms] {
    if {[$pin getSigType] in {POWER GROUND}} { continue }
    lassign [$pin getAvgXY] valid x y
    if {!$valid} { error "missing pin geometry" }
    puts $out [join [list pin "[$inst getName]/[[$pin getMTerm] getName]" $x $y 0 0 [$pin getIoType]] "\t"]
  }
}
foreach row [$block getRows] {
  set box [$row getBBox]
  set site [$row getSite]
  puts $out [join [list row [$row getName] [$box xMin] [$box yMin] [expr {[$box xMax]-[$box xMin]}] [expr {[$box yMax]-[$box yMin]}] [$site getWidth]] "\t"]
}
foreach blockage [$block getBlockages] {
  set box [$blockage getBBox]
  puts $out [join [list blockage [$blockage getId] [$box xMin] [$box yMin] [expr {[$box xMax]-[$box xMin]}] [expr {[$box yMax]-[$box yMin]}] hard] "\t"]
}
close $out
'''

FEATURES = r'''
set result [open $::env(ENG_PIN_FEATURES) w]
puts $result "pin\tslew_rise_ns\tslew_fall_ns\tcapacitance_pf"
set scene [lindex [sta::get_scenes *] 0]
foreach pin [get_pins -hierarchical *] {
  set lib [$pin liberty_port]
  if {$lib eq "NULL"} { continue }
  set vertex [lindex [$pin vertices] 0]
  if {$vertex eq "NULL" || $vertex eq ""} { continue }
  puts $result [join [list [get_full_name $pin] \
    [sta::time_sta_ui [$vertex slew rise max]] [sta::time_sta_ui [$vertex slew fall max]] \
    [expr {[$lib capacitance $scene max] * 1e12}]] "\t"]
}
close $result

# All endpoint worst paths, separately for min/max and both transitions. This
# is an endpoint-worst-path inventory, NOT a claim to enumerate every graph path.
set result [open $::env(ENG_PATH_FEATURES) w]
puts $result "analysis\tendpoint\tlaunch_clock_sink\tcapture_clock_sink\tslack_ns"
set clock_sinks [dict create]
set clocks [open $::env(ENG_CLOCK_TABLE) r]
gets $clocks header
while {[gets $clocks row] >= 0} { dict set clock_sinks [lindex [split $row "\t"] 2] 1 }
close $clocks
foreach analysis {max min} {
  foreach edge {rise fall} {
    foreach pe [find_timing_paths -path_delay ${analysis}_${edge} -endpoint_path_count 1 -group_path_count 10000000 -unconstrained] {
      set launch ""
      set capture ""
      foreach pin [[$pe path] pins] {
        set name [get_full_name $pin]
        if {[dict exists $clock_sinks $name]} { set launch $name; break }
      }
      set target [$pe target_clk_path]
      if {$target ne "NULL"} {
        set name [get_full_name [$target pin]]
        if {[dict exists $clock_sinks $name]} { set capture $name }
      }
      puts $result [join [list $analysis [get_full_name [$pe pin]] $launch $capture [sta::time_sta_ui [$pe slack]]] "\t"]
    }
  }
}
close $result
'''


def main():
    p = argparse.ArgumentParser()
    for key in ('flow', 'platform', 'bottom_odb', 'top_odb', 'measurement', 'output'):
        p.add_argument('--' + key.replace('_', '-'), required=True)
    args = p.parse_args()
    sys.path.insert(0, args.flow)
    from open3dbench.backend.three_d_opt import _run_openroad, _platform_files
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    for die in ('bottom', 'top'):
        _, libs = _platform_files(Path(args.platform), die)
        _run_openroad(GEOMETRY, {'ENG_ODB': getattr(args, die + '_odb'),
            'ENG_LIBS': '\n'.join(map(str, libs)), 'ENG_GEOMETRY': str(output / (die + '-geometry.tsv'))},
            output / (die + '-geometry.log'), '/usr/local/bin/openroad')
    measurement = Path(args.measurement)
    # The original generated script uses absolute in-container artifact paths.
    # Host launcher maps the measurement back to that same /out/measurement path.
    script = (measurement / 'assembly/passive/validate_sta.tcl').read_text() + '\n' + FEATURES
    _run_openroad(script, {'ENG_PIN_FEATURES': str(output / 'pins.tsv'),
        'ENG_PATH_FEATURES': str(output / 'paths.tsv'),
        'ENG_CLOCK_TABLE': str(measurement / 'assembly/passive/clock_sinks_by_clock.tsv')},
        output / 'features.log', '/usr/local/bin/openroad')
    print('Native local-task feature extraction completed')


if __name__ == '__main__':
    main()
