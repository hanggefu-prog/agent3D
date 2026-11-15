export DESIGN_NICKNAME = bp_be
export DESIGN_NAME = bp_be_top
include $(DESIGN_HOME)/asap7/open3dbench_common.mk

export VERILOG_FILES = ./designs/src/$(DESIGN_NAME)/pickled.v \
                       ./designs/nangate45/$(DESIGN_NAME)/macros.v \
                       ./designs/asap7/fakeram45_to_fakeram7.v
export SDC_FILE = ./designs/asap7/$(DESIGN_NAME)/bp_be_top.sdc
export ADDITIONAL_LEFS = $(addprefix $(PLATFORM_DIR)/lef/,fakeram7_512x64.lef fakeram7_64x15.lef fakeram7_64x96.lef)
export ADDITIONAL_LIBS = $(addprefix $(PLATFORM_DIR)/lib/NLDM/,fakeram7_512x64.lib fakeram7_64x15.lib fakeram7_64x96.lib)
export DIE_AREA = 0 0 350 300
export CORE_AREA = 5 5 345 295
