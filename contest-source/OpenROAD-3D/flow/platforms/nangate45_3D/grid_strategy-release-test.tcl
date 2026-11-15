####################################
# global connections
####################################
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDD$} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDPE$}
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDCE$}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSS$} -ground
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSSE$}
global_connect
####################################
# voltage domains
####################################
set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}
####################################
# standard cell grid
####################################
if {![info exists ::env(OPEN3D_DIE)]} {
  error "OPEN3D_DIE must be bottom or top"
}
if {$::env(OPEN3D_DIE) eq "bottom"} {
  set rail_layer metal1
  set mesh_layer metal4
  set mesh_pitch 10.0
  set trunk_layer metal7
  # Keep the same-net M7 spacing below the height of the smallest SRAM PG
  # window.  With 30um pitch only the alternating VDD stripe crossed a
  # fakeram45_64x7 instance in valid MoL placements, isolating its VSS grid.
  set trunk_pitch 20.0
  set trunk_offset 2.0
  set macro_pin_layer metal4
} elseif {$::env(OPEN3D_DIE) eq "top"} {
  # The top cell rail is mirrored onto M19.  M16/M13 form an orthogonal
  # mesh while fakeram PG pins on M17 connect down without crossing the
  # macro obstructions on M18-M20.
  set rail_layer metal19
  set mesh_layer metal16
  # Legalized standard cells can occupy short row fragments beside a macro
  # halo.  A 2um M16 pitch keeps both supplies reachable in those fragments;
  # the former 4um pitch left method-dependent islands in otherwise legal
  # movable-HBT placements.
  set mesh_pitch 2.0
  set trunk_layer metal13
  set trunk_pitch 20.0
  set trunk_offset 1.0
  set macro_pin_layer metal17
} else {
  error "invalid OPEN3D_DIE=$::env(OPEN3D_DIE)"
}
define_pdn_grid -name {grid} -voltage_domains {CORE} -pins [list $trunk_layer]
# The rail must follow the actual standard-cell rows.  A plain pitched stripe
# can miss the PG pins after row cutting around macros, leaving every cell
# electrically disconnected even though the higher-layer mesh is present.
add_pdn_stripe -grid {grid} -layer $rail_layer -width {0.17} -pitch {2.4} -offset {0} -followpins
# External 3-D placers can create short row fragments beside a macro keepout.
# The top mesh uses a 2um VDD/VSS pair pitch so a legal fragment wider than
# one standard-cell site reaches both supplies; the conventional bottom stack
# retains the lighter 10um mesh.
add_pdn_stripe -grid {grid} -layer $mesh_layer -width {0.48} -pitch $mesh_pitch -offset {0.5}
add_pdn_stripe -grid {grid} -layer $trunk_layer -width {1.40} -pitch $trunk_pitch -offset $trunk_offset
if {$::env(OPEN3D_DIE) eq "bottom"} {
  add_pdn_connect -grid {grid} -layers [list $rail_layer $mesh_layer]
  add_pdn_connect -grid {grid} -layers [list $mesh_layer $trunk_layer]
} else {
  # PDNGEN layer pairs are always specified from the lower numbered routing
  # layer to the higher one, even though the top die's physical stack is
  # used in the opposite direction.
  add_pdn_connect -grid {grid} -layers [list $mesh_layer $rail_layer]
  # A single stacked via is sufficient at each dense M13/M16 intersection and
  # avoids proposing a four-via array that must be ripped up beside local
  # obstructions.  The 2um mesh provides many redundant intersections.
  add_pdn_connect -grid {grid} -layers [list $trunk_layer $mesh_layer] \
    -max_rows 1 -max_columns 1
}
####################################
# macro grid
####################################
# Nangate45 fakeram VDD/VSS pins are long stripes.  A 90-degree macro
# rotation also rotates those pins, so normal and rotated instances require
# orthogonal first-hop stripes just like the native Nangate45 PDN strategy.
if {$::env(OPEN3D_DIE) eq "bottom"} {
  define_pdn_grid -name {macro_compact} -voltage_domains {CORE} -macro \
    -orient {R0 R180 MX MY} -halo {2.0 2.0 2.0 2.0} \
    -cells {fakeram45_64x7.*} -grid_over_pg_pins
  add_pdn_stripe -grid {macro_compact} -layer {metal5} -width {0.93} \
    -pitch {12.0} -offset {3.0} -extend_to_boundary
  add_pdn_stripe -grid {macro_compact} -layer {metal6} -width {0.48} \
    -pitch {5.0} -offset {0.5} -extend_to_boundary
  add_pdn_connect -grid {macro_compact} -layers {metal4 metal5}
  add_pdn_connect -grid {macro_compact} -layers {metal5 metal6}
  add_pdn_connect -grid {macro_compact} -layers {metal6 metal7}

  define_pdn_grid -name {macro_normal} -voltage_domains {CORE} -macro \
    -orient {R0 R180 MX MY} -halo {2.0 2.0 2.0 2.0} \
    -cells {^fakeram45_(?!64x7_).*} -grid_over_pg_pins
  add_pdn_stripe -grid {macro_normal} -layer {metal5} -width {0.93} -pitch {10.0} -offset {2}
  add_pdn_stripe -grid {macro_normal} -layer {metal6} -width {0.93} -pitch {10.0} -offset {2}
  add_pdn_connect -grid {macro_normal} -layers {metal4 metal5}
  add_pdn_connect -grid {macro_normal} -layers {metal5 metal6}
  add_pdn_connect -grid {macro_normal} -layers {metal6 metal7}

  define_pdn_grid -name {macro_rotated} -voltage_domains {CORE} -macro \
    -orient {R90 R270 MXR90 MYR90} -halo {2.0 2.0 2.0 2.0} \
    -cells {fakeram.*} -grid_over_pg_pins
  add_pdn_stripe -grid {macro_rotated} -layer {metal6} -width {0.93} -pitch {10.0} -offset {2}
  add_pdn_connect -grid {macro_rotated} -layers {metal4 metal6}
  add_pdn_connect -grid {macro_rotated} -layers {metal6 metal7}
} else {
  # The standard-cell/tapcell keepout is 10um, but the macro-grid obstruction
  # halo must remain local to the macro.  DREAMPlace may legally place macro
  # boundaries adjacent; a 10um PDN halo would make neighboring instance grids
  # erase each other's straps.  Two microns matches the native Nangate policy.
  define_pdn_grid -name {macro_compact} -voltage_domains {CORE} -macro \
    -orient {R0 R180 MX MY} -halo {2.0 2.0 2.0 2.0} \
    -cells {fakeram45_64x7.*} -grid_over_boundary -reversed_stack
  add_pdn_stripe -grid {macro_compact} -layer {metal16} -width {0.28} \
    -number_of_straps {1} -pitch {4.48} -spacing {1.96} -offset {2.66} -starts_with GROUND \
    -extend_to_boundary
  add_pdn_stripe -grid {macro_compact} -layer {metal15} -width {0.93} \
    -number_of_straps {1} -pitch {12.0} -spacing {5.0} -offset {2.0} -extend_to_boundary
  add_pdn_stripe -grid {macro_compact} -layer {metal14} -width {0.48} \
    -number_of_straps {1} -pitch {5.0} -spacing {2.0} -offset {0.5} -extend_to_boundary
  add_pdn_connect -grid {macro_compact} -layers {metal16 metal17} \
    -max_rows 1 -max_columns 1
  add_pdn_connect -grid {macro_compact} -layers {metal15 metal16}
  add_pdn_connect -grid {macro_compact} -layers {metal14 metal15}
  add_pdn_connect -grid {macro_compact} -layers {metal13 metal14}

  define_pdn_grid -name {macro_normal} -voltage_domains {CORE} -macro \
    -orient {R0 R180 MX MY} -halo {2.0 2.0 2.0 2.0} \
    -cells {^fakeram45_(?!64x7_).*} -grid_over_boundary -reversed_stack
  # The fakeram PG stripes are 0.28um wide, with alternating VSS/VDD centers
  # 2.24um apart.  A narrow, centered M16 stripe avoids the neighboring M17
  # obstruction while retaining one legal via per PG stripe.
  add_pdn_stripe -grid {macro_normal} -layer {metal16} -width {0.14} \
    -pitch {4.48} -spacing {2.10} -offset {2.73} -starts_with GROUND \
    -extend_to_boundary
  add_pdn_stripe -grid {macro_normal} -layer {metal15} -width {0.93} \
    -pitch {10.0} -offset {2} -extend_to_boundary
  add_pdn_stripe -grid {macro_normal} -layer {metal14} -width {0.93} \
    -pitch {10.0} -offset {2} -extend_to_boundary
  add_pdn_connect -grid {macro_normal} -layers {metal16 metal17} \
    -max_rows 1 -max_columns 1
  add_pdn_connect -grid {macro_normal} -layers {metal15 metal16}
  add_pdn_connect -grid {macro_normal} -layers {metal14 metal15}
  add_pdn_connect -grid {macro_normal} -layers {metal13 metal14}

  define_pdn_grid -name {macro_rotated} -voltage_domains {CORE} -macro \
    -orient {R90 R270 MXR90 MYR90} -halo {2.0 2.0 2.0 2.0} \
    -cells {fakeram.*} -grid_over_boundary -reversed_stack
  add_pdn_stripe -grid {macro_rotated} -layer {metal16} -width {0.93} \
    -pitch {10.0} -offset {2} -extend_to_boundary
  add_pdn_stripe -grid {macro_rotated} -layer {metal15} -width {0.93} \
    -pitch {10.0} -offset {2} -extend_to_boundary
  add_pdn_stripe -grid {macro_rotated} -layer {metal14} -width {0.93} \
    -pitch {10.0} -offset {2} -extend_to_boundary
  add_pdn_connect -grid {macro_rotated} -layers {metal16 metal17} \
    -max_rows 1 -max_columns 1
  add_pdn_connect -grid {macro_rotated} -layers {metal15 metal16}
  add_pdn_connect -grid {macro_rotated} -layers {metal14 metal15}
  add_pdn_connect -grid {macro_rotated} -layers {metal13 metal14}
}
