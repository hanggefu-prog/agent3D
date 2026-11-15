# Open3DBench 3D synthesis frontend for Ariane133.
#
# Reuse the released RTL, SRAM wrappers, timing constraints, and NG45 macro
# views, but do not inherit its 2D RTLMP/hierarchical implementation policy.
include $(DESIGN_HOME)/nangate45/ariane133/config.mk

# The SDC period is 4ns.  Yosys/ABC's -D argument is explicitly in ps; the
# generic ORFS fallback copies the bare SDC value and would otherwise pass 4.
export ABC_CLOCK_PERIOD_IN_PS = 4000

# Open3DBench owns partitioning, floorplanning, and macro placement after
# synthesis.  Flatten standard-cell logic so ABC can optimize paths across
# RTL module boundaries.  FakeRAM remains a hard macro through the included
# LEF/Liberty and Verilog wrapper contract.
export SYNTH_HIERARCHICAL = 0
export RTLMP_FLOW = 0
export FLOW_VARIANT = 3d_frontend
