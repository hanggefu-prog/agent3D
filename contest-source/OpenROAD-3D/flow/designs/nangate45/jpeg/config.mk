export DESIGN_NICKNAME = jpeg
export DESIGN_NAME = jpeg_encoder
export PLATFORM = nangate45

export SYNTH_HIERARCHICAL = 1
export VERILOG_FILES = $(sort $(wildcard $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/*.v))
export VERILOG_INCLUDE_DIRS = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/include
export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/jpeg_encoder.sdc
export ABC_CLOCK_PERIOD_IN_PS = 1000
export ABC_AREA = 1

# Registered source 2-D outline.  The main 3-D flow later refloorplans each
# tier to the catalogued 300 x 300 um footprint.
export DIE_AREA = 0 0 450 450
export CORE_AREA = 10 12 440 438

export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
