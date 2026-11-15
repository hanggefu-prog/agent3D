export DESIGN_NAME = bsg_chip
export DESIGN_NICKNAME = bp_quad
include $(DESIGN_HOME)/asap7/open3dbench_common.mk

export VERILOG_FILES = ./designs/src/$(DESIGN_NICKNAME)/bsg_chip_block.sv2v.v \
                       ./designs/nangate45/$(DESIGN_NICKNAME)/macros.v \
                       ./designs/asap7/fakeram45_to_fakeram7.v
export SDC_FILE = ./designs/asap7/$(DESIGN_NICKNAME)/bsg_chip.sdc
export ADDITIONAL_LEFS = $(addprefix $(PLATFORM_DIR)/lef/,fakeram7_256x48.lef fakeram7_32x32.lef fakeram7_64x124.lef fakeram7_512x64.lef fakeram7_64x62.lef fakeram7_128x116.lef)
export ADDITIONAL_LIBS = $(addprefix $(PLATFORM_DIR)/lib/NLDM/,fakeram7_256x48.lib fakeram7_32x32.lib fakeram7_64x124.lib fakeram7_512x64.lib fakeram7_64x62.lib fakeram7_128x116.lib)
export DIE_AREA = 0 0 700 700
export CORE_AREA = 5 5 695 695
