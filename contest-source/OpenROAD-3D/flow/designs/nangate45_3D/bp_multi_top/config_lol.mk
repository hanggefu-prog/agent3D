export DESIGN_NICKNAME = bp_multi
export DESIGN_NAME = bp_multi_top
export PLATFORM    = nangate45_3D

export SYNTH_HIERARCHICAL = 1
export FLOW_VARIANT = lol
export PLACE_3D_ROOT ?= ../../Place-3D
export INPUT_DEF = $(PLACE_3D_ROOT)/binaries/converted_output/${DEF_VARIANT}/${METHOD}/bp_multi.def
export IDEAL_CLOCK = 1

export VERILOG_FILES = ./designs/src/$(DESIGN_NAME)/pickled.v \
                       ./designs/$(PLATFORM)/$(DESIGN_NAME)/macros.v
export SDC_FILE      = ./designs/$(PLATFORM)/$(DESIGN_NAME)/bp_multi_top.sdc
export ABC_AREA      = 1

export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef_top/fakeram45_512x64.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_256x96.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_32x64.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_64x7.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_64x15.top.lef \
                         $(PLATFORM_DIR)/lef_top/fakeram45_64x96.top.lef \
                         $(PLATFORM_DIR)/lef_top/NangateOpenCellLibrary.macro.mod.top.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_512x64.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_256x96.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_32x64.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_64x7.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_64x15.bottom.lef \
                         $(PLATFORM_DIR)/lef_bottom/fakeram45_64x96.bottom.lef \
                
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib_top/fakeram45_512x64.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_256x96.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_32x64.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_64x7.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_64x15.top.lib \
                         $(PLATFORM_DIR)/lib_top/fakeram45_64x96.top.lib \
                         $(PLATFORM_DIR)/lib_top/NangateOpenCellLibrary_typical.top.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_512x64.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_256x96.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_32x64.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_64x7.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_64x15.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/fakeram45_64x96.bottom.lib \
                         $(PLATFORM_DIR)/lib_bottom/NangateOpenCellLibrary_typical.bottom.lib 

export DIE_AREA    = 0 0 800 800 
export CORE_AREA   = 0 0 800 800 

export MACRO_PLACE_HALO = 10 10
export MACRO_PLACE_CHANNEL = 20 20

export PLACE_DENSITY_LB_ADDON = 0.05
export SKIP_GATE_CLONING      = 1

export DETAILED_ROUTE_ARGS = -droute_end_iter 5
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 5
