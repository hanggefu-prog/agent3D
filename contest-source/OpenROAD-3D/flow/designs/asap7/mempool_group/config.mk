export DESIGN_NAME = mempool_group
export DESIGN_NICKNAME = mempool_group
export PLATFORM    = asap7

export SYNTH_HIERARCHICAL = 1

export TEMP_DESIGN_DIR = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/axi/src/axi_pkg.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/cf_math_pkg.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/riscv_instr.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_pkg.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/mempool_pkg.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/cluster_interconnect/rtl/tcdm_interconnect/tcdm_interconnect_pkg.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/cluster_interconnect/rtl/variable_latency_interconnect/variable_latency_interconnect.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/axi_hier_interco.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/mempool_tile.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/axi/src/axi_mux.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/axi/src/axi_id_remap.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/mempool_cc.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_icache/snitch_icache_pkg.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_icache/snitch_icache.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/tcdm_adapter.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/fakeram45_256x32.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/fakeram45_64x64.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/tech_cells_generic/src/rtl/tc_sram.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/spill_register.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/fall_through_register.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/stream_xbar.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/address_scrambler.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/tcdm_shim.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_demux.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_axi_adapter.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/axi/src/axi_cut.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/axi/src/axi_id_prepend.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/rr_arb_tree.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/fifo_v3.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/lzc.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_ipu.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/isochronous_spill_register.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_icache/snitch_icache_lookup.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_icache/snitch_icache_l0.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/stream_arbiter.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_icache/snitch_icache_refill.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/onehot_to_bin.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/stream_demux.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/deprecated/fifo_v2.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/spill_register_flushable.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/stream_arbiter_flushable.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/latch_scm.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_lsu.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/tech_cells_generic/src/rtl/tc_clk.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_icache/snitch_icache_handler.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch_addr_demux.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_regfile_ff.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_shared_muldiv.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/common_cells/src/deprecated/find_first_one.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_onehot.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/snitch/src/snitch_icache/snitch_icache_lfsr.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/axi/src/axi_intf.sv \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/mempool_group.sv

export VERILOG_INCLUDE_DIRS = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl \
	$(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/rtl/register_interface/include

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/$(DESIGN_NAME).sdc
export ABC_CLOCK_PERIOD_IN_PS = 3000

export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_256x32.lef \
                         $(PLATFORM_DIR)/lef/fakeram45_64x64.lef

export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/NLDM/fakeram45_256x32.lib \
                         $(PLATFORM_DIR)/lib/NLDM/fakeram45_64x64.lib

export DIE_AREA    = 0 0 300 300
export CORE_AREA   = 2 2 298 298

export CORNER = TC
export DFF_LIB_FILE = $(PLATFORM_DIR)/lib/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib
export DONT_USE_SC_LIB = $(OBJECTS_DIR)/lib/merged.lib
export SC_LEF = $(PLATFORM_DIR)/lef/asap7sc7p5t_28_R_1x_220121a.dreamplace.lef
export MACRO_PLACE_HALO = 2 2
export MACRO_PLACE_CHANNEL = 4 4
export PLACE_DENSITY = 0.50
export TNS_END_PERCENT = 100
export NUM_CORES ?= 16

export SYNTH_HDL_FRONTEND = slang
export SYNTH_BLACKBOXES = fakeram45_256x32 fakeram45_64x64

export DETAILED_ROUTE_ARGS = -droute_end_iter 0
export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 2
