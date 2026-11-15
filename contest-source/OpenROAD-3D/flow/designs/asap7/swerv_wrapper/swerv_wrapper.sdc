current_design swerv_wrapper
set_cmd_units -time ns
set clk_name core_clock 
set clk_port_name clk
set clk_period 2.0
set clk_io_pct 0.2

set clk_port [get_ports -quiet $clk_port_name]
set jtag_clk_port [get_ports -quiet jtag_tck]

if {[llength $clk_port] != 0} {
  create_clock -name $clk_name -period $clk_period $clk_port
}
if {[llength $jtag_clk_port] != 0} {
  create_clock -name jtag_clock -period $clk_period $jtag_clk_port
}
if {[llength [get_clocks -quiet $clk_name]] != 0
    && [llength [get_clocks -quiet jtag_clock]] != 0} {
  set_clock_groups -asynchronous -group $clk_name -group jtag_clock
}

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]
set non_clock_inputs [lsearch -inline -all -not -exact $non_clock_inputs $jtag_clk_port]

if {[llength [get_clocks -quiet $clk_name]] != 0} {
  set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
  set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]
}
set jtag_tdo_port [get_ports -quiet jtag_tdo]
if {[llength [get_clocks -quiet jtag_clock]] != 0 && [llength $jtag_tdo_port] != 0} {
  set_output_delay [expr $clk_period * $clk_io_pct] -clock jtag_clock $jtag_tdo_port
}
