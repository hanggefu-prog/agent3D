# Native ten-layer PDN used independently in both die databases.
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDD$} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDPE$}
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDCE$}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSS$} -ground
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSSE$}
global_connect
set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}

if {![info exists ::env(OPEN3D_DIE)] || $::env(OPEN3D_DIE) ni {bottom top}} {
  error "OPEN3D_DIE must be bottom or top"
}

set rail_layer metal1
set mesh_layer metal4
set trunk_layer metal7
define_pdn_grid -name {grid} -voltage_domains {CORE} -pins [list $trunk_layer]
add_pdn_stripe -grid {grid} -layer $rail_layer -width {0.17} -pitch {2.4} -offset {0} -followpins
add_pdn_stripe -grid {grid} -layer $mesh_layer -width {0.48} -pitch {10.0} -offset {0.5}
add_pdn_stripe -grid {grid} -layer $trunk_layer -width {1.40} -pitch {20.0} -offset {2.0}
add_pdn_connect -grid {grid} -layers [list $rail_layer $mesh_layer]
add_pdn_connect -grid {grid} -layers [list $mesh_layer $trunk_layer]

# Fakeram PG pins are on local M4.  Normal and rotated macros use the same
# native access grids on both dies; assembly mirroring never changes layers.
define_pdn_grid -name {macro_compact} -voltage_domains {CORE} -macro \
  -orient {R0 R180 MX MY} -halo {2.0 2.0 2.0 2.0} \
  -cells {fakeram45_64x(?:7|15|21).*} -grid_over_pg_pins
add_pdn_stripe -grid {macro_compact} -layer {metal5} -width {0.93} \
  -pitch {12.0} -offset {3.0} -extend_to_boundary
add_pdn_stripe -grid {macro_compact} -layer {metal6} -width {0.48} \
  -pitch {5.0} -offset {0.5} -extend_to_boundary
add_pdn_connect -grid {macro_compact} -layers {metal4 metal5}
add_pdn_connect -grid {macro_compact} -layers {metal5 metal6}
add_pdn_connect -grid {macro_compact} -layers {metal6 metal7}

define_pdn_grid -name {macro_normal} -voltage_domains {CORE} -macro \
  -orient {R0 R180 MX MY} -halo {2.0 2.0 2.0 2.0} \
  -cells {^(?:fakeram45_(?!64x(?:7|15|21)_)|mem(?:_0)?_ext_).*} -grid_over_pg_pins
add_pdn_stripe -grid {macro_normal} -layer {metal5} -width {0.93} -pitch {10.0} -offset {2}
add_pdn_stripe -grid {macro_normal} -layer {metal6} -width {0.93} -pitch {10.0} -offset {2}
add_pdn_connect -grid {macro_normal} -layers {metal4 metal5}
add_pdn_connect -grid {macro_normal} -layers {metal5 metal6}
add_pdn_connect -grid {macro_normal} -layers {metal6 metal7}

define_pdn_grid -name {macro_rotated} -voltage_domains {CORE} -macro \
  -orient {R90 R270 MXR90 MYR90} -halo {2.0 2.0 2.0 2.0} \
  -cells {^(?:fakeram|mem_).*} -grid_over_pg_pins
add_pdn_stripe -grid {macro_rotated} -layer {metal6} -width {0.93} -pitch {10.0} -offset {2}
add_pdn_connect -grid {macro_rotated} -layers {metal4 metal6}
add_pdn_connect -grid {macro_rotated} -layers {metal6 metal7}

# Dense legal macro placements can leave channels too narrow for PDNGen to
# add an extra repair strap.  Keep those channels as warnings; the flow's
# subsequent check_power_grid calls remain the authoritative fatal gate for
# VDD and VSS connectivity on each die.
pdn::allow_repair_channels true
