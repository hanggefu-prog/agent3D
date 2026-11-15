export DESIGN_NICKNAME = ibex
export DESIGN_NAME = ibex_core
export PLATFORM = nangate45

export SYNTH_HIERARCHICAL = 1
export VERILOG_FILES = $(sort $(wildcard $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/*.v))
export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/ibex_core.sdc
export ABC_CLOCK_PERIOD_IN_PS = 2200

# Registered source 2-D outline.  The main 3-D flow later refloorplans each
# tier to the catalogued 150 x 150 um footprint.
export DIE_AREA = 0 0 225 225
export CORE_AREA = 10 12 215 213

export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
