export DESIGN_NICKNAME = jpeg
export DESIGN_NAME = jpeg_encoder
export PLATFORM = asap7

export SYNTH_HIERARCHICAL = 1
export VERILOG_FILES = $(sort $(wildcard $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/*.v))
export VERILOG_INCLUDE_DIRS = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/include
export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/jpeg_encoder.sdc
export ABC_CLOCK_PERIOD_IN_PS = 1000
export ABC_AREA = 1

# Registered ASAP7 source outline.  The main flow refloorplans each tier to
# the catalogued 90 x 80 um footprint.
export DIE_AREA = 0 0 120 120
export CORE_AREA = 2 2 118 118

export CORNER = TC
export DFF_LIB_FILE = $(PLATFORM_DIR)/lib/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib
export DONT_USE_SC_LIB = $(OBJECTS_DIR)/lib/merged.lib
export SC_LEF = $(PLATFORM_DIR)/lef/asap7sc7p5t_28_R_1x_220121a.dreamplace.lef
export PLACE_DENSITY = 0.55
export TNS_END_PERCENT = 100
export NUM_CORES ?= 16

export DETAILED_ROUTE_ARGS = -droute_end_iter 0
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 2
