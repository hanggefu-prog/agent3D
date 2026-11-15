"""Accept functional-latch diagnostics only with live constrained-path proof."""
import math
from pathlib import Path
import re


LATCH_PROOF_TCL = r"""
set latch_proof [open $::env(REPORT_LATCH_PROOF) w]
foreach gate [all_registers -clock_pins -level_sensitive] {
  if {[llength [get_property $gate clocks]] != 0} { continue }
  set gate_name [open3d_object_full_name $gate]
  set cell_name [open3d_pin_instance_name $gate_name]
  foreach data [all_registers -data_pins -level_sensitive] {
    set data_name [open3d_object_full_name $data]
    if {[open3d_pin_instance_name $data_name] ne $cell_name} { continue }
    set slacks {}
    foreach delay {min max} {
      # Without -unconstrained, an absent/false-pathed data cone yields no path.
      set paths [find_timing_paths -to $data -path_delay $delay -group_path_count 1]
      if {[llength $paths] == 0} { error "functional latch $data_name has no constrained $delay path" }
      set path [lindex $paths 0]
      if {![$path is_path_delay] || [$path is_unconstrained]} {
        error "functional latch $data_name lacks an explicit $delay delay bound"
      }
      if {[open3d_object_full_name [get_property $path endpoint]] ne $data_name} {
        error "functional latch timing proof has another endpoint"
      }
      lappend slacks [get_property $path slack]
    }
    puts $latch_proof "$gate_name\t$data_name\t[join $slacks \t]"
  }
}
close $latch_proof
"""


def retain_data_checks(script):
    # Disabling a latch enable also removes its setup/hold endpoint in OpenSTA,
    # defeating the explicit max/min-delay constraints. Keep these checks in
    # this scoring view and prove every acknowledged data endpoint below.
    old = "    set_disable_timing $unclocked_latch_pins"
    if script.count(old) != 1:
        raise ValueError("Functional-latch scoring adapter requires requalification")
    return script.replace(old, "    # Keep explicit bounded data checks for scoring.").replace(
        "model=bounded_data_disable_enable_checks", "model=bounded_data_checks_retained")


def audit(check_setup, proof):
    clocks, endpoints = set(), set()
    for line in Path(proof).read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError("Malformed functional-latch proof")
        clock, data = fields[:2]
        if clock.rsplit("/", 1)[0] != data.rsplit("/", 1)[0] or data in endpoints:
            raise ValueError("Mismatched/duplicate functional-latch proof")
        if any(not math.isfinite(float(x)) or abs(float(x)) >= 1e20 for x in fields[2:]):
            raise ValueError("Functional latch lacks finite min/max timing")
        clocks.add(clock)
        endpoints.add(data)
    warnings = []
    current = None
    patterns = ((r"Warning: There (?:are|is) (\d+) unclocked register/latch pins?\.", clocks),
                (r"Warning: There (?:are|is) (\d+) unconstrained endpoints?\.", endpoints))
    for line in Path(check_setup).read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("Warning:"):
            match = next(((re.fullmatch(pattern, line), allowed) for pattern, allowed in patterns
                          if re.fullmatch(pattern, line)), None)
            if match is None:
                raise ValueError("Unexpected check_setup warning: " + line)
            found, allowed = match
            current = {"count": int(found.group(1)), "pins": [], "allowed": allowed}
            warnings.append(current)
        elif current is not None and line.startswith("  "):
            current["pins"].append(line.strip())
        else:
            raise ValueError("Unrecognized check_setup evidence")
    for row in warnings:
        pins = row["pins"]
        if len(pins) != row["count"] or len(set(pins)) != len(pins) or not set(pins) <= row["allowed"]:
            raise ValueError("Unproven check_setup latch diagnostic")
    return {"functional_latches": len(clocks), "bounded_data_pins": len(endpoints),
            "acknowledged_warnings": len(warnings)}
