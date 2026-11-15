set sdc_version 2.0

# Set the current design
current_design ariane

create_clock -name "core_clock" -period 4.0 -waveform {0.0 2.0} [get_ports clk_i]

set_input_delay 0 -clock core_clock [all_inputs -no_clocks]
set_output_delay 0 -clock core_clock [all_outputs]
set_false_path -from [get_ports -quiet rst_ni]
# Ariane contains four data-enabled functional latches.  Their enables are not
# periodic clocks.  Constrain their data cones to one core period instead of
# inventing clock waveforms or dropping the paths as false.
set latch_data [all_registers -data_pins -level_sensitive]
if {[llength $latch_data] > 0} {
  set_max_delay 4.0 -to $latch_data
  set_min_delay 0.0 -to $latch_data
}
unset latch_data
