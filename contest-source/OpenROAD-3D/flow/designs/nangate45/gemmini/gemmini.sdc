set sdc_version 2.0
set_units -time ns -resistance kOhm -capacitance fF -power mW -voltage V -current uA

create_clock -name clock -period 3.0 [get_ports clock]
set_clock_uncertainty 0.06 [get_clocks clock]
set_clock_latency 0.07 [get_clocks clock]
set_input_delay -clock clock 0.6 [all_inputs -no_clocks]
set_output_delay -clock clock 0.6 [all_outputs]
set_false_path -from [get_ports reset]
set_max_transition 0.30 [current_design]
set_max_fanout 16 [current_design]
