export DESIGN_NAME = ariane
export DESIGN_NICKNAME = ariane136
include $(DESIGN_HOME)/asap7/open3dbench_common.mk

export VERILOG_FILES = ./designs/src/$(DESIGN_NICKNAME)/ariane.sv2v.v \
                       ./designs/nangate45/$(DESIGN_NICKNAME)/macros.v \
                       ./designs/asap7/fakeram45_to_fakeram7.v
export SDC_FILE = ./designs/asap7/$(DESIGN_NICKNAME)/ariane.sdc
export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram7_256x16.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/NLDM/fakeram7_256x16.lib
export DIE_AREA = 0 0 270 270
export CORE_AREA = 5 5 265 265
