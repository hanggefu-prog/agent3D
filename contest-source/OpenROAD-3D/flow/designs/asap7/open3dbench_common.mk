# Shared, PDK-only settings for Open3DBench ASAP7 RTL frontends.
export PLATFORM = asap7
export SYNTH_HIERARCHICAL = 0
export RTLMP_FLOW = 0
export CORNER = TC
export DFF_LIB_FILE = $(PLATFORM_DIR)/lib/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib
export DONT_USE_SC_LIB = $(OBJECTS_DIR)/lib/merged.lib
export SC_LEF = $(PLATFORM_DIR)/lef/asap7sc7p5t_28_R_1x_220121a.dreamplace.lef
export MACRO_PLACE_HALO = 2 2
export MACRO_PLACE_CHANNEL = 4 4
export PLACE_DENSITY = 0.55
export TNS_END_PERCENT = 100
export DETAILED_ROUTE_ARGS = -droute_end_iter 0
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 2
