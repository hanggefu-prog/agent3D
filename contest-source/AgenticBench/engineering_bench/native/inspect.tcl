# Host-owned read-back. Names/roles are obtained from ODB + Liberty + real SDC.
# No layout/netlist edits and no evaluator status is inferred from this alone.
set_thread_count 2
read_db $::env(ENG_ODB)
foreach lib [split $::env(ENG_LIBS) "\n"] {
  if {$lib ne ""} { read_liberty $lib }
}
set_cmd_units -time ns
read_sdc $::env(ENG_SDC)
set block [ord::get_db_block]
set clock_nets [dict create]
foreach net [sta::find_all_clk_nets] { dict set clock_nets [$net getName] 1 }
set cells [open $::env(ENG_OUTPUT)/cells.tsv w]
set pins [open $::env(ENG_OUTPUT)/pins.tsv w]
set ports [open $::env(ENG_OUTPUT)/ports.tsv w]
puts $cells "name\tmaster\tx_dbu\ty_dbu\torient\tstatus\tmaster_type\tclock_cell\tcts_owned\tbuffer\tinverter"
puts $pins "instance\tpin\tnet\tdirection\tsignal_type"
puts $ports "port\tnet\tdirection\tsignal_type\tlayer\txmin\tymin\txmax\tymax\tstatus"
set master_cache [dict create]
set cell_count 0
set pg_count 0
set floating_pg 0
foreach inst [$block getInsts] {
  set master [$inst getMaster]
  set master_name [$master getName]
  if {![dict exists $master_cache $master_name]} {
    set libcells [get_lib_cells -quiet */$master_name]
    set buffer 0
    set inverter 0
    if {[llength $libcells] == 1} {
      set libcell [lindex $libcells 0]
      set buffer [$libcell is_buffer]
      set inverter [$libcell is_inverter]
    } elseif {[llength $libcells] > 1} {
      error "ambiguous Liberty master $master_name"
    }
    dict set master_cache $master_name [list $buffer $inverter]
  }
  lassign [dict get $master_cache $master_name] buffer inverter
  set ownership [odb::dbIntProperty_find $inst OPENROAD_CTS_BUFFER]
  set owned [expr {$ownership ne "NULL" && [$ownership getValue] != 0}]
  set input_count 0
  set output_count 0
  set all_clock 1
  foreach pin [$inst getITerms] {
    set net [$pin getNet]
    set net_name ""
    if {$net ne "NULL"} { set net_name [$net getName] }
    set signal_type [$pin getSigType]
    puts $pins [join [list [$inst getName] [[$pin getMTerm] getName] $net_name [$pin getIoType] $signal_type] "\t"]
    if {$signal_type in {POWER GROUND}} {
      incr pg_count
      if {$net eq "NULL"} { incr floating_pg }
      continue
    }
    if {[$pin getIoType] eq "INPUT"} { incr input_count }
    if {[$pin getIoType] eq "OUTPUT"} { incr output_count }
    # TritonCTS load-balancing cells intentionally leave the output open.
    # Require persistent native ownership + a Liberty buffer/inverter, not a
    # clkload name or SOURCE TIMING heuristic, for this special case.
    if {$net eq "NULL" && [$pin getIoType] eq "OUTPUT" && $owned && ($buffer || $inverter)} { continue }
    if {![dict exists $clock_nets $net_name]} { set all_clock 0 }
  }
  # A clock gate is NOT removable buffering. A master name/SOURCE tag is not
  # sufficient; role needs proven one-input/one-output function and real clocks.
  set clock_cell [expr {($buffer || $inverter) && $input_count == 1 && $output_count == 1 && $all_clock}]
  lassign [$inst getLocation] x y
  puts $cells [join [list [$inst getName] $master_name $x $y [$inst getOrient] \
    [$inst getPlacementStatus] [$master getType] $clock_cell $owned $buffer $inverter] "\t"]
  incr cell_count
}
foreach port [$block getBTerms] {
  set net [$port getNet]
  set net_name ""
  if {$net ne "NULL"} { set net_name [$net getName] }
  set count 0
  foreach pin [$port getBPins] {
    foreach box [$pin getBoxes] {
      set layer [$box getTechLayer]
      if {$layer eq "NULL"} { continue }
      puts $ports [join [list [$port getName] $net_name [$port getIoType] [$port getSigType] \
        [$layer getName] [$box xMin] [$box yMin] [$box xMax] [$box yMax] [$pin getPlacementStatus]] "\t"]
      incr count
    }
  }
  if {$count == 0} {
    puts $ports [join [list [$port getName] $net_name [$port getIoType] [$port getSigType] "" "" "" "" "" ""] "\t"]
  }
}
close $cells
close $pins
close $ports
puts "ENGINEERING_NATIVE_SNAPSHOT cells=$cell_count clocks=[llength [all_clocks]] pg=$pg_count floating_pg=$floating_pg"
