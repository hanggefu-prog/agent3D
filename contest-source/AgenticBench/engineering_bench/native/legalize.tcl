# T4 protected normalization, fixed policy v1. Candidate code cannot edit this.
set_thread_count 8
read_db $::env(ENG_ODB)
set block [ord::get_db_block]
set input [open $::env(ENG_RESTORE) r]
set restored 0
while {[gets $input line] >= 0} {
  if {$line eq ""} { continue }
  lassign [split $line "\t"] name master x y orient status fixed
  set inst [$block findInst $name]
  if {$inst eq "NULL"} { error "original data instance missing: $name" }
  if {[[$inst getMaster] getName] ne $master} { error "original data master changed: $name" }
  $inst setPlacementStatus PLACED
  $inst setOrient $orient
  $inst setLocation $x $y
  if {$fixed} {
    $inst setPlacementStatus FIRM
  } else {
    $inst setPlacementStatus $status
  }
  incr restored
}
close $input
# Retain candidate clock-cell placement. The data-cell initialization and this
# exact standard legalizer are shared by native baseline and all candidates.
detailed_placement -max_displacement {500 500}
check_placement -verbose
set pg_count 0
foreach inst [$block getInsts] {
  foreach pin [$inst getITerms] {
    set type [[$pin getMTerm] getSigType]
    if {$type ni {POWER GROUND}} { continue }
    set net [$pin getNet]
    set expected [expr {$type eq "POWER" ? "VDD" : "VSS"}]
    if {$net eq "NULL" || [$net getName] ne $expected} {
      error "invalid PG connection: [$inst getName]/[[$pin getMTerm] getName]"
    }
    incr pg_count
  }
}
write_db $::env(ENG_OUTPUT)/design.odb
write_def $::env(ENG_OUTPUT)/design.def
puts "ENGINEERING_PROTECTED_LEGALIZATION restored=$restored pg=$pg_count"
