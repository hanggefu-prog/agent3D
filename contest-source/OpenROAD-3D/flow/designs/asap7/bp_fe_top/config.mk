export DESIGN_NICKNAME = bp_fe
export DESIGN_NAME = bp_fe_top
export PLATFORM = asap7

# Keep the same RTL hierarchy and macro boundary as the ng45 benchmark.  The
# three memories below are regenerated with the upstream ASAP7 FakeRAM model.
export SYNTH_HIERARCHICAL = 0
export RTLMP_FLOW = 0
export RTLMP_MAX_INST = 30000
export RTLMP_MIN_INST = 5000
export RTLMP_MAX_MACRO = 12
export RTLMP_MIN_MACRO = 4

export VERILOG_FILES = ./designs/src/$(DESIGN_NAME)/pickled.v \
                       ./designs/$(PLATFORM)/$(DESIGN_NAME)/macros.v
export SDC_FILE = ./designs/$(PLATFORM)/$(DESIGN_NAME)/bp_fe_top.sdc

export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram7_512x64.lef \
                         $(PLATFORM_DIR)/lef/fakeram7_64x7.lef \
                         $(PLATFORM_DIR)/lef/fakeram7_64x96.lef
export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/NLDM/fakeram7_512x64.lib \
                         $(PLATFORM_DIR)/lib/NLDM/fakeram7_64x7.lib \
                         $(PLATFORM_DIR)/lib/NLDM/fakeram7_64x96.lib

# This is the source 2-D synthesis/floorplan canvas.  The maintained 3-D flow
# projects it into the explicit 200x160um per-die footprint.
export DIE_AREA = 0 0 300 240
export CORE_AREA = 5 5 295 235
export CORNER = TC
# The Open3DBench flow Makefile predates ORFS's per-corner DFF selector.  Bind
# the same RVT/TT sequential library explicitly for dfflibmap.
export DFF_LIB_FILE = $(PLATFORM_DIR)/lib/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib
# This Open3DBench ORFS snapshot feeds one Liberty path to ABC.  ASAP7 splits
# combinational cells across AO/INVBUF/OA/SIMPLE/SEQ files, so select the
# existing deterministic merge target instead of silently mapping against the
# first (AO-only) file.
export DONT_USE_SC_LIB = $(OBJECTS_DIR)/lib/merged.lib
# DREAMPlace treats ASAP7's CORE TIEHIGH/TIELOW LEF subclasses as fixed
# physical cells.  Use a generated view that changes only those synthesized
# tie cells to plain CORE; the upstream physical LEF remains untouched.
export SC_LEF = $(PLATFORM_DIR)/lef/asap7sc7p5t_28_R_1x_220121a.dreamplace.lef

export MACRO_PLACE_HALO = 2 2
export MACRO_PLACE_CHANNEL = 4 4
export PLACE_DENSITY = 0.55
export TNS_END_PERCENT = 100

# The unified 3-D runner owns final routing; the frontend stops at its clean
# post-floorplan checkpoint.
export DETAILED_ROUTE_ARGS = -droute_end_iter 0
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 2
