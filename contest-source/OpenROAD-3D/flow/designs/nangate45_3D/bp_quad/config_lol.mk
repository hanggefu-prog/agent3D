export DESIGN_NAME = bp_quad
export DESIGN_NICKNAME = bp_quad
export PLATFORM    = nangate45_3D

export SYNTH_HIERARCHICAL = 1
export FLOW_VARIANT = lol
export PLACE_3D_ROOT ?= ../../Place-3D
export INPUT_DEF = $(PLACE_3D_ROOT)/binaries/converted_output/${DEF_VARIANT}/${METHOD}/bp_quad.def
export IDEAL_CLOCK = 1

export VERILOG_FILES = ./designs/src/$(DESIGN_NICKNAME)/bsg_chip_block.sv2v.v \
                       ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/macros.v

export SDC_FILE      = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/bsg_chip.sdc

export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef_top/fakeram45_256x48.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_32x32.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_64x124.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_512x64.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_64x62.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_128x116.top.lef \
                         $(PLATFORM_DIR)/lef_top/NangateOpenCellLibrary.macro.mod.top.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_256x48.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_32x32.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_64x124.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_512x64.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_64x62.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_128x116.bottom.lef
                
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib_top/fakeram45_256x48.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_32x32.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_64x124.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_512x64.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_64x62.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_128x116.top.lib \
                         $(PLATFORM_DIR)/lib_top/NangateOpenCellLibrary_typical.top.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_256x48.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_32x32.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_64x124.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_512x64.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_64x62.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_128x116.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/NangateOpenCellLibrary_typical.bottom.lib 

export DIE_AREA    = 0 0 2500 2500
export CORE_AREA   = 0 0 2500 2500

export MACRO_PLACE_HALO = 10 10
export MACRO_PLACE_CHANNEL = 20 20

export DETAILED_ROUTE_ARGS = -droute_end_iter 5
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 5
