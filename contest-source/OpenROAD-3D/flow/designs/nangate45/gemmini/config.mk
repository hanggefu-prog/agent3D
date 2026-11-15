export DESIGN_NAME = Gemmini
export DESIGN_NICKNAME = gemmini
export PLATFORM = nangate45

export SYNTH_HIERARCHICAL = 1
export RTLMP_FLOW = 0
export GEMMINI_RTL_DIR = $(DESIGN_HOME)/src/gemmini/rtl
export VERILOG_FILES = $(sort $(wildcard $(GEMMINI_RTL_DIR)/*.sv) $(wildcard $(GEMMINI_RTL_DIR)/*.v))
export VERILOG_INCLUDE_DIRS = $(GEMMINI_RTL_DIR)
export SYNTH_HDL_FRONTEND = slang
export SYNTH_BLACKBOXES = mem_ext mem_0_ext

export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/gemmini/gemmini.sdc
export ABC_CLOCK_PERIOD_IN_PS = 3000

export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/mem_ext.lef \
                         $(PLATFORM_DIR)/lef/mem_0_ext.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/mem_ext.lib \
                         $(PLATFORM_DIR)/lib/mem_0_ext.lib

# Deterministic source 2-D canvas.  The registered 3-D footprint is 2200x2200
# um per tier, close to half of this source area.
export DIE_AREA = 0 0 3200 3000
export CORE_AREA = 10 10 3190 2990
export MACRO_PLACE_HALO = 10 10
export MACRO_PLACE_CHANNEL = 20 20
export PLACE_DENSITY = 0.52
export TNS_END_PERCENT = 100
export NUM_CORES ?= 16

export DETAILED_ROUTE_ARGS = -droute_end_iter 0
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 2
