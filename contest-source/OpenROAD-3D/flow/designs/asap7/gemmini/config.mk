export DESIGN_NAME = Gemmini
export DESIGN_NICKNAME = gemmini
export PLATFORM = asap7

export SYNTH_HIERARCHICAL = 1
export RTLMP_FLOW = 0
export GEMMINI_RTL_DIR = $(DESIGN_HOME)/src/gemmini/rtl
export VERILOG_FILES = $(sort $(wildcard $(GEMMINI_RTL_DIR)/*.sv) $(wildcard $(GEMMINI_RTL_DIR)/*.v))
export VERILOG_INCLUDE_DIRS = $(GEMMINI_RTL_DIR)
export SYNTH_HDL_FRONTEND = slang
export SYNTH_BLACKBOXES = mem_ext mem_0_ext

export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/gemmini/gemmini.sdc
export ABC_CLOCK_PERIOD_IN_PS = 2000

export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/mem_ext.lef \
                         $(PLATFORM_DIR)/lef/mem_0_ext.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/NLDM/mem_ext.lib \
                         $(PLATFORM_DIR)/lib/NLDM/mem_0_ext.lib

export DIE_AREA = 0 0 900 800
export CORE_AREA = 5 5 895 795
export CORNER = TC
export DFF_LIB_FILE = $(PLATFORM_DIR)/lib/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib
export DONT_USE_SC_LIB = $(OBJECTS_DIR)/lib/merged.lib
export SC_LEF = $(PLATFORM_DIR)/lef/asap7sc7p5t_28_R_1x_220121a.dreamplace.lef
export MACRO_PLACE_HALO = 2 2
export MACRO_PLACE_CHANNEL = 4 4
export PLACE_DENSITY = 0.50
export TNS_END_PERCENT = 100
export NUM_CORES ?= 16

export DETAILED_ROUTE_ARGS = -droute_end_iter 0
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 2
