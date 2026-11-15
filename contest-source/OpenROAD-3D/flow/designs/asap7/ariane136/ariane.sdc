set sdc_version 2.0
set_cmd_units -time ns
current_design ariane

create_clock -name core_clock -period 3.0 -waveform {0.0 3.0} [get_ports clk_i]
set_input_delay 0 -clock core_clock [all_inputs -no_clocks]
set_output_delay 0 -clock core_clock [all_outputs]
set_false_path -from [get_ports -quiet rst_ni]
# These are data-enabled functional latches, not periodic clock domains.
set_max_delay 3.0 -to [all_registers -data_pins -level_sensitive]
set_min_delay 0.0 -to [all_registers -data_pins -level_sensitive]
