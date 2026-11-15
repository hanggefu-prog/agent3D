export DESIGN_NAME = swerv_wrapper
include $(DESIGN_HOME)/asap7/open3dbench_common.mk

export VERILOG_FILES = ./designs/src/swerv/swerv_wrapper.sv2v.v \
                       ./designs/nangate45/swerv_wrapper/macros.v \
                       ./designs/asap7/fakeram45_to_fakeram7.v
export SDC_FILE = ./designs/asap7/swerv_wrapper/swerv_wrapper.sdc
export ADDITIONAL_LEFS = $(addprefix $(PLATFORM_DIR)/lef/,fakeram7_2048x39.lef fakeram7_256x34.lef fakeram7_64x21.lef)
export ADDITIONAL_LIBS = $(addprefix $(PLATFORM_DIR)/lib/NLDM/,fakeram7_2048x39.lib fakeram7_256x34.lib fakeram7_64x21.lib)
export DIE_AREA = 0 0 450 400
export CORE_AREA = 5 5 445 395
