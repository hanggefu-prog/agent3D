# ASAP7 Liberty is expressed in ps.  Normalize the command interface to ns so
# Gemmini uses the same human-readable constraints as the ng45 frontend.
set_cmd_units -time ns

create_clock -name clock -period 2.0 [get_ports clock]
set_clock_uncertainty 0.04 [get_clocks clock]
set_clock_latency 0.07 [get_clocks clock]
set_input_delay -clock clock 0.4 [all_inputs -no_clocks]
set_output_delay -clock clock 0.4 [all_outputs]
set_false_path -from [get_ports reset]
set_max_transition 0.32 [current_design]
set_max_fanout 16 [current_design]
