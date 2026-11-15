# To remove [WARNING STA-1212] from the logs for ASAP7.
# /OpenROAD-flow-scripts/flow/platforms/asap7/lib/asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib.gz
#   line 13178, timing group from output port.
# Added following suppress_message
if {[llength [info commands suppress_message]]} { suppress_message STA 1212 }
# Physical-only ASAP7 tapcells have LEF but no timing arcs.
if {[llength [info commands suppress_message]]} { suppress_message ORD 2011 }
