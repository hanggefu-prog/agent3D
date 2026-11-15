"""Passive-HBT-aware CTS for the maintained two-die physical model.

The two OpenDB databases remain the physical sources of truth.  A receiving-die
preview is embedded in the continuous passive assembly so near/far cutpoint
queries can derive the exact capacitance and insertion-delay proxy required by
source-die TritonCTS.  The same passive query derives the receiving clock
arrival and slew after source CTS.  The temporary local proxy is therefore a
physical CTS aid only; fixed HBT delay and directional timing adapters are not
timing authorities.  The intermediate SPEF is only a serialization of
OpenROAD's placement estimator; it is not extracted RC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from ..command_log import copy_with_log_reference
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .contracts import ContractError, validate_checkpoint
from .clock_crossing import ClockCrossingError, clock_ports_from_sdc_text, literal_clock_port_query
from .die_sdc import DieSDCError, _commands, _edits, _group_end, _query
from .hbt_materialize import (
    _read_boundary_net_map,
    _repair_boundary_verilog,
    _validate_boundary_verilog,
    _validate_netlist_liberty,
)
from .opt_handoff import HBTBoundary, OptHandoff, load_cts_handoff
from .passive_hbt import (
    passive_links,
    stitch_estimated_spef,
    write_passive_assembly,
    write_passive_plan,
)
from .pg_connect import PG_CONNECT_TCL
from .sdc_handoff import PORT_INVENTORY_TCL, project_sdc_for_handoff, write_sdc_projection
from .sta_completeness import STA_COMPLETENESS_TCL
from .three_d_opt import (
    PLACEMENT_DUMP_TCL,
    RESIZER_BUFFERING_WARNINGS,
    _assembly_rows,
    _environment_script,
    _platform_files,
    _read_key_values,
    _rebuild_assembly,
    _run_openroad,
    _warning_summary,
    _write_json,
)


class ThreeDCTSError(RuntimeError):
    """Raised when CTS cannot preserve or validate the two-die model."""


GENERATED_SDC_MARKER = "# Open3DBench generated HBT boundary constraints."
LOAD_PREFIX = "HBT_CLK_TX_LOAD_"


def _clock_boundaries(handoff: OptHandoff) -> Tuple[HBTBoundary, ...]:
    clocks = tuple(row for row in handoff.boundaries if row.is_clock)
    if handoff.ideal_pre_cts_ports:
        raise ThreeDCTSError(
            "CTS refuses ideal cross-die clock ports {}; rerun 50_hbt so every "
            "receiving-die clock has a formal clock HBT".format(
                ", ".join(handoff.ideal_pre_cts_ports)
            )
        )
    if not clocks:
        raise ThreeDCTSError("CTS requires at least one formal clock HBT")
    physical_port = _physical_clock_port(_clean_sdc(handoff.dies["bottom"].sdc))
    primaries = tuple(
        row
        for row in clocks
        if physical_port in {row.net_name, row.physical_net}
    )
    if len(primaries) != 1:
        raise ThreeDCTSError(
            "the clock-domain contract requires exactly one formal HBT for the "
            "physical clock port {}; found {}".format(
                physical_port, len(primaries)
            )
        )
    clock = primaries[0]
    if clock.source_die != "bottom" or clock.sink_die != "top":
        raise ThreeDCTSError(
            "the maintained all-IO-on-bottom policy requires a bottom-to-top "
            "clock HBT, got {}".format(clock.direction)
        )
    return (clock,) + tuple(row for row in clocks if row is not clock)


def _clean_sdc(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.split(GENERATED_SDC_MARKER, 1)[0].rstrip() + "\n"


def _strip_io_delay_commands(text: str) -> str:
    """Remove complete Tcl I/O-delay commands, including continuations."""

    kept: List[str] = []
    command: List[str] = []
    for line in text.splitlines(keepends=True):
        command.append(line)
        if line.rstrip().endswith("\\"):
            continue
        joined = "".join(command)
        if not re.match(r"^\s*set_(?:input|output)_delay\b", joined):
            kept.append(joined)
        command = []
    if command:
        joined = "".join(command)
        if not re.match(r"^\s*set_(?:input|output)_delay\b", joined):
            kept.append(joined)
    return "".join(kept)


def _physical_clock_name(text: str) -> str:
    """Return the named clock attached to the single physical clock port."""

    normalized = re.sub(r"\\\s*\n", " ", text)
    for command in re.findall(r"(?:^|\n)\s*create_clock\b[^\n]*", normalized):
        name = re.search(
            r"\s-name\s+(?:\{([^}]+)\}|\"([^\"]+)\"|([^\s]+))", command
        )
        if name:
            value = next(group for group in name.groups() if group is not None)
            if value.startswith("$"):
                variable = re.escape(value[1:])
                assignment = re.search(
                    r"^\s*set\s+{}\s+(?:\{{([^}}]+)\}}|\"([^\"]+)\"|([^\s]+))\s*$".format(
                        variable
                    ),
                    normalized,
                    re.MULTILINE,
                )
                if not assignment:
                    raise ThreeDCTSError(
                        "cannot resolve physical clock name variable {}".format(value)
                    )
                value = next(
                    group for group in assignment.groups() if group is not None
                )
            return value
        try:
            return clock_ports_from_sdc_text(command)[0]
        except ClockCrossingError:
            continue
    raise ThreeDCTSError("cannot identify named physical clock in SDC")


def _physical_clock_port(text: str) -> str:
    """Return the primary physical clock port.

    The first declared physical clock is the performance/coordination clock.
    Additional physical clocks are still built by local CTS and represented
    as auxiliary formal HBT clock roots; the continuous passive assembly is
    their final timing authority.
    """

    try:
        ports = clock_ports_from_sdc_text(text)
    except ClockCrossingError as exc:
        raise ThreeDCTSError(
            "CTS requires at least one physical create_clock/get_ports root"
        ) from exc
    return ports[0]


def _replace_physical_clock_port(text: str, old: str, new: str) -> str:
    """Move one primary clock collection from an assembly IO to an HBT port."""

    # The shared die-SDC projector emits canonical literal Tcl lists. Keep
    # this form valid without reverting to a regex that stops at the inner ].
    for match in re.finditer(r"\[get_ports(?:\s+-quiet)?\s+\[list\b", text):
        try:
            end = _group_end(text, match.start(), "[", "]")
            ports = literal_clock_port_query(text[match.start():end])
        except (DieSDCError, ClockCrossingError) as exc:
            raise ThreeDCTSError("unsupported physical clock literal collection") from exc
        if old in ports:
            return text[:match.start()] + _query([new if port == old else port for port in ports]) + text[end:]
    direct = re.compile(
        r"(get_ports(?:\s+-[A-Za-z_][A-Za-z0-9_]*)*\s+)"
        r"\{?" + re.escape(old) + r"\}?([\]\s])"
    )
    text, count = direct.subn(
        lambda match: match.group(1) + "{" + new + "}" + match.group(2),
        text,
        count=1,
    )
    if count:
        return text

    # Resolve an indirect collection such as:
    #   set clk_port_name clk
    #   set clk_port [get_ports $clk_port_name]
    collection_vars = re.findall(
        r"\[get_ports(?:\s+-[A-Za-z_][A-Za-z0-9_]*)*\s+"
        r"\$([A-Za-z_][A-Za-z0-9_]*)\]",
        text,
    )
    for variable in collection_vars:
        assignment = re.compile(
            r"(?m)^(\s*set\s+" + re.escape(variable) + r"\s+)"
            r"(?:\{" + re.escape(old) + r"\}|\"" + re.escape(old)
            + r"\"|" + re.escape(old) + r")(\s*)$"
        )
        text, count = assignment.subn(
            lambda match: match.group(1) + new + match.group(2), text, count=1
        )
        if count:
            return text
    raise ThreeDCTSError(
        "cannot replace physical clock port {} with {}".format(old, new)
    )


def _append_incoming_clock_roots(
    text: str,
    die: str,
    clocks: Sequence[HBTBoundary],
    primary: HBTBoundary,
    period_ns: float,
    transition_ns: float,
) -> str:
    """Constrain derived clocks which enter a die through formal HBTs.

    These local roots are CTS construction aids.  The continuous passive
    assembly remains the timing authority and preserves the real relationship
    through the remote clock gate and HBT RC.
    """

    incoming = [
        row for row in clocks if row is not primary and row.sink_die == die
    ]
    if not incoming:
        return text
    lines = ["", "# Open3DBench derived cross-die clock roots."]
    for row in incoming:
        name = "OPEN3D_" + re.sub(r"[^A-Za-z0-9_]", "_", row.hbt_id)
        lines.extend(
            (
                "create_clock -name {{{}}} -period {:.9f} [get_ports {{{}}}]".format(
                    name, period_ns, row.sink_port
                ),
                "set_clock_latency -source 0.000000000 [get_clocks {{{}}}]".format(
                    name
                ),
                "set_clock_transition {:.9f} [get_clocks {{{}}}]".format(
                    transition_ns, name
                ),
                "set_input_transition {:.9f} [get_ports {{{}}}]".format(
                    transition_ns, row.sink_port
                ),
            )
        )
    return text.rstrip() + "\n" + "\n".join(lines) + "\n"


def _bind_receiving_clock_guard(text: str, sink_port: str) -> str:
    """Bind one projected primary-clock existence guard to its CTS HBT root.

    OPT deliberately keeps local presence queries empty on the remote die,
    while mapping create_clock to a proven clock HBT. CTS consumes that
    already-projected SDC. Only the single-command positive existence guard
    around this literal HBT clock may change; the collection variable and all
    other guards retain their original local-only meaning.
    """
    def literal_ports(raw: str):
        try:
            return literal_clock_port_query(raw)
        except ClockCrossingError:
            # An unrelated scalar/dynamic collection is not evidence for
            # this narrowly supported guard. The shared projector still
            # validates its semantics at the physical consumer.
            return None

    def primary_commands(script: str):
        found = []
        for start, end, words in _commands(script):
            if words[0].raw == "create_clock":
                roots = []
                for word in words[1:]:
                    if word.raw.startswith("[get_ports"):
                        roots.extend(literal_ports(word.raw) or ())
                if sink_port in roots:
                    if roots != [sink_port]:
                        raise ThreeDCTSError("receiving primary clock must have exactly one HBT root")
                    found.append((start, end))
            elif words[0].raw == "if":
                for word in words[2:]:
                    if word.raw.startswith("{"):
                        found.extend(primary_commands(word.raw[1:-1]))
        return found

    try:
        declarations = primary_commands(text)
        if not declarations:
            return text
        if len(declarations) > 1:
            raise ThreeDCTSError("multiple receiving primary clock declarations are unsupported")
        empty_collections = set()
        bound_collections = set()
        edits = []
        for _start, _end, words in _commands(text):
            command = words[0].raw
            if command == "set" and len(words) == 3:
                name = words[1].raw
                empty_collections.discard(name)
                bound_collections.discard(name)
                if re.fullmatch(r"\[list\s*\]", words[2].raw):
                    empty_collections.add(name)
                elif words[2].raw.startswith("[get_ports") and literal_ports(words[2].raw) == [sink_port]:
                    bound_collections.add(name)
            elif command == "unset":
                empty_collections.difference_update(word.raw for word in words[1:])
                bound_collections.difference_update(word.raw for word in words[1:])
            elif command == "if":
                if not any(primary_commands(word.raw[1:-1]) for word in words[2:] if word.raw.startswith("{")):
                    # A conditional assignment cannot establish a stable empty
                    # collection for a later receiving-clock guard.
                    for name in empty_collections | bound_collections:
                        if re.search(r"\b(?:set|unset)\s+" + re.escape(name) + r"(?:\s|$)", text[_start:_end]):
                            empty_collections.discard(name)
                            bound_collections.discard(name)
                    continue
                body_index = 3 if len(words) > 2 and words[2].raw == "then" else 2
                if len(words) != body_index + 1 or not words[1].raw.startswith("{") or not words[body_index].raw.startswith("{"):
                    raise ThreeDCTSError("unsupported receiving primary clock guard")
                body = _commands(words[body_index].raw[1:-1])
                if len(body) != 1 or body[0][2][0].raw != "create_clock":
                    raise ThreeDCTSError("receiving primary clock guard must contain only its create_clock")
                condition = words[1]
                guard = re.fullmatch(
                    r"\s*\[llength\s+(.+)\]\s*(?:>\s*0|!=\s*0)\s*",
                    condition.raw[1:-1],
                )
                operand = guard.group(1).strip() if guard else ""
                variable = re.fullmatch(r"\$(?:([A-Za-z_][A-Za-z_0-9:]*)|\{([A-Za-z_][A-Za-z_0-9:]*)\})", operand)
                name = (variable.group(1) or variable.group(2)) if variable else None
                if name in bound_collections or (operand.startswith("[get_ports") and literal_ports(operand) == [sink_port]):
                    continue
                if name not in empty_collections:
                    raise ThreeDCTSError("unsupported or unproven receiving primary clock existence guard")
                start = condition.start + 1 + guard.start(1)
                end = condition.start + 1 + guard.end(1)
                edits.append((start, end, _query([sink_port])))
        return _edits(text, edits)
    except (DieSDCError, ClockCrossingError) as exc:
        raise ThreeDCTSError("unsupported receiving primary clock Tcl: {}".format(exc)) from exc


def _receiving_sdc(
    path: Path,
    clock: HBTBoundary,
    source_latency_ns: float,
    transition_ns: float,
    clocks: Sequence[HBTBoundary] = (),
) -> str:
    """Move the original top-level clock from the absent IO to the HBT root."""

    text = _clean_sdc(path)
    clock_name = _physical_clock_name(text)
    # The maintained IO policy places every original top-level IO on bottom.
    # Keeping those commands in the top SDC produces hundreds of predictable
    # missing-port warnings and hides real constraint mistakes.  Cross-die data
    # timing is checked in the assembly view, so the top CTS view only keeps
    # clock and non-IO constraints.
    text = _strip_io_delay_commands(text)
    text = _replace_physical_clock_port(
        text, _physical_clock_port(text), clock.sink_port
    )
    text = _bind_receiving_clock_guard(text, clock.sink_port)
    text += (
        "\n# Open3DBench CTS HBT-root model.\n"
        "set_clock_latency -source {latency:.9f} [get_clocks {{{clock_name}}}]\n"
        "set_clock_transition {transition:.9f} [get_clocks {{{clock_name}}}]\n"
        "set_input_transition {transition:.9f} [get_ports {{{sink_port}}}]\n"
    ).format(
        latency=source_latency_ns,
        transition=transition_ns,
        sink_port=clock.sink_port,
        clock_name=clock_name,
    )
    return _append_incoming_clock_roots(
        text,
        clock.sink_die,
        clocks,
        clock,
        _clock_period_ns(path),
        transition_ns,
    )


def _source_sdc(
    path: Path,
    clocks: Sequence[HBTBoundary] = (),
    transition_ns: float = 0.02,
) -> str:
    if not clocks:
        return _clean_sdc(path)
    return _append_incoming_clock_roots(
        _clean_sdc(path),
        "bottom",
        clocks,
        clocks[0],
        _clock_period_ns(path),
        transition_ns,
    )


def _clock_period_ns(path: Path) -> float:
    text = _clean_sdc(path)
    variable = re.search(r"^\s*set\s+clk_period\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    if variable:
        period = float(variable.group(1))
    else:
        literal = re.search(r"\bcreate_clock\b[^\n]*\s-period\s+([0-9.eE+-]+)", text)
        if not literal:
            raise ThreeDCTSError("cannot determine clock period from {}".format(path))
        period = float(literal.group(1))
    if not math.isfinite(period) or period <= 0.0:
        raise ThreeDCTSError("invalid clock period in {}".format(path))
    return period


def _safe_cell_name(hbt_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", hbt_id)
    return LOAD_PREFIX + name


def _write_clock_load_models(
    liberty: Path,
    lef: Path,
    clock: HBTBoundary,
    capacitance_pf: float,
    downstream_min_ns: float,
    downstream_max_ns: float,
    landing_layer: str = "metal10",
) -> str:
    if capacitance_pf <= 0.0 or downstream_min_ns < 0.0:
        raise ThreeDCTSError("invalid clock-HBT load model")
    if downstream_max_ns < downstream_min_ns:
        raise ThreeDCTSError("clock-HBT max insertion delay is smaller than min")
    cell = _safe_cell_name(clock.hbt_id)
    liberty.write_text(
        """library (open3dbench_clock_hbt) {{
  delay_model : table_lookup ;
  time_unit : \"1ns\" ;
  capacitive_load_unit (1,pf) ;
  cell ({cell}) {{
    area : 0.0 ;
    interface_timing : true ;
    pin (CLK) {{
      direction : input ;
      clock : true ;
      capacitance : {cap:.9f} ;
      timing () {{
        timing_sense : positive_unate ;
        timing_type : min_clock_tree_path ;
        cell_rise (scalar) {{ values (\"{dmin:.9f}\") ; }}
        cell_fall (scalar) {{ values (\"{dmin:.9f}\") ; }}
      }}
      timing () {{
        timing_sense : positive_unate ;
        timing_type : max_clock_tree_path ;
        cell_rise (scalar) {{ values (\"{dmax:.9f}\") ; }}
        cell_fall (scalar) {{ values (\"{dmax:.9f}\") ; }}
      }}
    }}
    pin (DUMMY) {{
      direction : output ;
      function : "CLK" ;
      max_capacitance : 1.0 ;
    }}
  }}
}}
""".format(
            cell=cell,
            cap=capacitance_pf,
            dmin=downstream_min_ns,
            dmax=downstream_max_ns,
        ),
        encoding="utf-8",
    )
    lef.write_text(
        """VERSION 5.8 ;
BUSBITCHARS \"[]\" ;
DIVIDERCHAR \"/\" ;
MACRO {cell}
  CLASS COVER ;
  ORIGIN 0 0 ;
  FOREIGN {cell} 0 0 ;
  SIZE 0.5 BY 0.5 ;
  SYMMETRY X Y ;
  PIN CLK
    DIRECTION INPUT ;
    USE CLOCK ;
    PORT
      LAYER {layer} ;
      RECT 0 0 0.5 0.5 ;
    END
  END CLK
  PIN DUMMY
    DIRECTION OUTPUT ;
    USE CLOCK ;
    PORT
      LAYER {layer} ;
      RECT 0 0 0.5 0.5 ;
    END
  END DUMMY
END {cell}
END LIBRARY
""".format(cell=cell, layer=landing_layer),
        encoding="utf-8",
    )
    return cell


DIE_CTS_TCL = PORT_INVENTORY_TCL + STA_COMPLETENESS_TCL + PG_CONNECT_TCL + r"""
read_db $::env(CTS_INPUT_ODB)
foreach lib [split $::env(CTS_DIE_LIBS) "\n"] {
  if {$lib ne ""} { read_liberty $lib }
}
if {$::env(CTS_CLOCK_LOAD_LIB) ne ""} {
  read_lef $::env(CTS_CLOCK_LOAD_LEF)
  read_liberty $::env(CTS_CLOCK_LOAD_LIB)
}
open3d_validate_sdc_ports $::env(CTS_DIE_SDC).ports.tsv
set_cmd_units -time ns
read_sdc $::env(CTS_DIE_SDC)
source $::env(CTS_SET_RC)
set_wire_rc -signal -layer $::env(CTS_SIGNAL_LAYER)
set_wire_rc -clock -layer $::env(CTS_CLOCK_LAYER)

set block [ord::get_db_block]
# OPT protects multi-BTerm boundary nets from resizer buffering.  CTS must own
# the real clock net, so remove that protection from this one explicitly while
# leaving every data-HBT boundary untouched.
set clock_db_net [$block findNet $::env(CTS_CLOCK_NET)]
if {$clock_db_net eq "NULL"} { error "missing physical clock net $::env(CTS_CLOCK_NET)" }
$clock_db_net setDoNotTouch false
# A die-local HBT LANDING is a routing anchor, not a logical CTS sink.  OPT
# deliberately protects the COVER instance, so detach only the clock landing
# while CTS and clock repair reshape the logical clock tree.  The original net
# owns a boundary BTerm and therefore remains alive; reconnect the landing to
# that same net after all clock transforms.
set clock_hbt_anchor_iterms {}
foreach inst [$block getInsts] {
  set landing [$inst findITerm LANDING]
  if {$landing eq "NULL"} { continue }
  set landing_net [$landing getNet]
  if {$landing_net eq "NULL"} { continue }
  if {[$landing_net getName] ne [$clock_db_net getName]} { continue }
  $inst setDoNotTouch false
  $landing disconnect
  lappend clock_hbt_anchor_iterms $landing
}
set load_count 0
set clock_load_pin ""
set clock_load_iterm ""
if {$::env(CTS_CLOCK_LOAD_PLAN) ne ""} {
  set plan [open $::env(CTS_CLOCK_LOAD_PLAN) r]
  while {[gets $plan line] >= 0} {
    if {$line eq ""} { continue }
    lassign [split $line "\t"] instance master net_name x y
    set master_obj [[ord::get_db] findMaster $master]
    if {$master_obj eq "NULL"} { error "missing clock-HBT load master $master" }
    set net [$block findNet $net_name]
    if {$net eq "NULL"} { error "missing clock-HBT source net $net_name" }
    set inst [odb::dbInst_create $block $master_obj $instance]
    $inst setLocation $x $y
    $inst setPlacementStatus FIRM
    set pin [$inst findITerm CLK]
    if {$pin eq "NULL"} { error "clock-HBT load $instance has no CLK pin" }
    $pin connect $net
    set clock_load_iterm $pin
    set clock_load_pin [get_pins [$pin getName]]
    incr load_count
  }
  close $plan
}
if {$load_count != $::env(CTS_EXPECTED_CLOCK_LOADS)} {
  error "clock-HBT load count $load_count != $::env(CTS_EXPECTED_CLOCK_LOADS)"
}

repair_clock_inverters
# Long-wire repair must precede TritonCTS latency balancing.  In particular,
# the synthetic HBT leaf carries the measured receiving-tree insertion delay;
# repairing its route afterwards would invalidate the delay comparison used by
# the balancer.
set cts_args [list -root_buf $::env(CTS_BUFFER) \
  -buf_list $::env(CTS_BUFFER) -sink_clustering_enable -repair_clock_nets \
  -delay_buffer_derate $::env(CTS_DELAY_BUFFER_DERATE)]
if {$::env(CTS_CLUSTER_SIZE) > 0} {
  lappend cts_args -sink_clustering_size $::env(CTS_CLUSTER_SIZE)
}
if {$::env(CTS_CLUSTER_DIAMETER) > 0} {
  lappend cts_args -sink_clustering_max_diameter $::env(CTS_CLUSTER_DIAMETER)
}
clock_tree_synthesis {*}$cts_args
set_propagated_clock [all_clocks]
estimate_parasitics -placement
# The synthetic load makes TritonCTS build a real branch to the HBT site.  It
# must not survive as a second logical load.  Transfer the boundary BTerm and
# physical HBT anchor to that leaf net, then delete the adapter.  The exported
# die Verilog/SPEF therefore contains the actual HBT branch, not the model.
set clock_boundary_net $clock_db_net
if {$::env(CTS_DIE) eq "bottom" && $clock_load_iterm ne ""} {
  set clock_boundary_net [$clock_load_iterm getNet]
  if {$clock_boundary_net eq "NULL"} { error "clock-HBT load was disconnected by CTS" }
  foreach port_name [split $::env(CTS_HANDOFF_PORTS) "\n"] {
    if {$port_name eq ""} { continue }
    set boundary [$block findBTerm $port_name]
    if {$boundary eq "NULL"} { error "missing clock HBT boundary $port_name" }
    $boundary disconnect
    $boundary connect $clock_boundary_net
  }
}
foreach landing $clock_hbt_anchor_iterms {
  $landing connect $clock_boundary_net
  [$landing getInst] setDoNotTouch true
}
detailed_placement -max_displacement {500 500}
check_placement -verbose
estimate_parasitics -placement
lassign [open3d_connect_pg_iterms 1] pg_iterms pg_iterms_repaired floating_pg_iterms
if {$floating_pg_iterms != 0} {
  error "$::env(CTS_DIE) CTS left $floating_pg_iterms floating PG ITerms"
}

check_setup -verbose > $::env(CTS_CHECK_SETUP_REPORT)
open3d_check_sta_completeness "cts_$::env(CTS_DIE)" 0
report_checks -path_delay max -group_path_count 10 > $::env(CTS_SETUP_REPORT)
report_checks -path_delay min -group_path_count 10 > $::env(CTS_HOLD_REPORT)

proc open3d_number {object property} {
  if {[catch {set value [get_property $object $property]}]} { return "" }
  return $value
}
set sink_file [open $::env(CTS_SINKS_TSV) w]
set sink_count 0
foreach pin [all_registers -clock_pins] {
  puts $sink_file [join [list [get_name $pin] \
    [open3d_number $pin arrival_max_rise] [open3d_number $pin arrival_max_fall] \
    [open3d_number $pin arrival_min_rise] [open3d_number $pin arrival_min_fall]] "\t"]
  incr sink_count
}
close $sink_file

set handoff_file [open $::env(CTS_HANDOFF_TSV) w]
foreach port_name [split $::env(CTS_HANDOFF_PORTS) "\n"] {
  if {$port_name eq ""} { continue }
  set port [get_ports -quiet [list $port_name]]
  if {[llength $port] != 1} { error "cannot resolve clock handoff port $port_name" }
  set observation $port
  if {$clock_load_pin ne ""} { set observation $clock_load_pin }
  puts $handoff_file [join [list $port_name \
    [open3d_number $observation arrival_max_rise] [open3d_number $observation arrival_max_fall] \
    [open3d_number $observation slew_max_rise] [open3d_number $observation slew_max_fall]] "\t"]
}
close $handoff_file

set cts_buffer_count 0
foreach inst [$block getInsts] {
  set cts_owner [odb::dbIntProperty_find $inst OPENROAD_CTS_BUFFER]
  if {$cts_owner eq "NULL" || ![$cts_owner getValue]} { continue }
  # Dummy load cells belong to CTS too, but do not drive a tree branch.
  set output [$inst getFirstOutput]
  if {$output ne "NULL" && [$output getNet] ne "NULL"} {
    incr cts_buffer_count
  }
}
if {$sink_count > 1 && $cts_buffer_count == 0} {
  error "CTS left $sink_count register clock sinks without a clock tree"
}
set metrics [open $::env(CTS_METRICS_TSV) w]
puts $metrics "clock_count\t[llength [all_clocks]]"
puts $metrics "register_clock_sink_count\t$sink_count"
puts $metrics "clock_buffer_count\t$cts_buffer_count"
puts $metrics "clock_hbt_load_count\t$load_count"
puts $metrics "clock_hbt_anchor_count\t[llength $clock_hbt_anchor_iterms]"
puts $metrics "pg_iterms\t$pg_iterms"
puts $metrics "pg_iterms_repaired\t$pg_iterms_repaired"
puts $metrics "floating_pg_iterms\t$floating_pg_iterms"
puts $metrics "wns_setup\t[sta::time_sta_ui [sta::worst_slack_cmd max]]"
puts $metrics "tns_setup\t[sta::time_sta_ui [sta::total_negative_slack_cmd max]]"
puts $metrics "wns_hold\t[sta::time_sta_ui [sta::worst_slack_cmd min]]"
puts $metrics "tns_hold\t[sta::time_sta_ui [sta::total_negative_slack_cmd min]]"
set root_net [get_nets -quiet [list $::env(CTS_CLOCK_NET)]]
if {[llength $root_net] != 1} { error "cannot resolve CTS root net for capacitance" }
set root_drivers [sta::net_driver_pins $root_net]
if {[llength $root_drivers] != 1} { error "CTS root net must have exactly one driver" }
lassign [sta::find_pi_elmore [lindex $root_drivers 0] rise max] root_c1 root_rpi root_c2
puts $metrics "clock_root_cap_pf\t[expr {($root_c1 + $root_c2) * 1.0e12}]"
close $metrics

# Keep the synthetic pin through reporting so its slew/arrival can be used as
# the bottom-to-top handoff observation.  It has already transferred its
# leaf net to the boundary BTerm; remove it before exporting physical state.
if {$::env(CTS_DIE) eq "bottom" && $clock_load_iterm ne ""} {
  set load_inst [$clock_load_iterm getInst]
  $clock_load_iterm disconnect
  odb::dbInst_destroy $load_inst
  set clock_load_iterm ""
  set clock_load_pin ""
}
set boundary_map [open $::env(CTS_BOUNDARY_NETS) w]
set boundary_ports [open $::env(CTS_BOUNDARY_PORTS_FILE) r]
while {[gets $boundary_ports port] >= 0} {
  if {$port eq ""} { continue }
  set bterm [$block findBTerm $port]
  if {$bterm eq "NULL" || [$bterm getNet] eq "NULL"} {
    error "cannot export HBT boundary net for $port"
  }
  puts $boundary_map "$port\t[[$bterm getNet] getName]"
}
close $boundary_ports
close $boundary_map

write_db $::env(CTS_OUTPUT_ODB)
write_def $::env(CTS_OUTPUT_DEF)
write_verilog $::env(CTS_OUTPUT_VERILOG)
write_sdc $::env(CTS_OUTPUT_SDC)
puts "OPEN3D_CTS_DIE_DONE die=$::env(CTS_DIE) sinks=$sink_count buffers=$cts_buffer_count"
"""


def _finite_values(values: Iterable[str]) -> List[float]:
    result = []
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and abs(value) < 1.0e20:
            result.append(value)
    return result


def _sink_latency(
    path: Path, rise_origin_ns: float = 0.0, fall_origin_ns: float = 0.0
) -> Dict[str, float]:
    rise: List[float] = []
    fall: List[float] = []
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 5:
            raise ThreeDCTSError("invalid CTS sink row in {}".format(path))
        rise.extend(value - rise_origin_ns for value in _finite_values((fields[1], fields[3])))
        fall.extend(value - fall_origin_ns for value in _finite_values((fields[2], fields[4])))
        count += 1
    if not rise or not fall:
        raise ThreeDCTSError("CTS produced no finite register-clock arrivals")
    return {
        "sink_count": float(count),
        "min_rise_ns": min(rise),
        "max_rise_ns": max(rise),
        "min_fall_ns": min(fall),
        "max_fall_ns": max(fall),
        "min_ns": min(rise + fall),
        "max_ns": max(rise + fall),
    }


def _handoff_values(path: Path, expected_port: str) -> Dict[str, float]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise ThreeDCTSError("expected one clock handoff row, found {}".format(len(rows)))
    fields = rows[0].split("\t")
    if len(fields) != 5 or fields[0] != expected_port:
        raise ThreeDCTSError("invalid clock handoff row")
    values = _finite_values(fields[1:])
    if len(values) != 4:
        raise ThreeDCTSError("clock handoff has incomplete arrival/slew")
    return {
        "arrival_max_rise_ns": float(fields[1]),
        "arrival_max_fall_ns": float(fields[2]),
        "slew_max_rise_ns": float(fields[3]),
        "slew_max_fall_ns": float(fields[4]),
    }


def _boundary_rows(handoff: OptHandoff) -> List[Dict[str, object]]:
    return [
        {
            "source_die": row.source_die,
            "sink_die": row.sink_die,
            "source_port": row.source_port,
            "sink_port": row.sink_port,
            "physical_net": row.physical_net or row.net_name,
        }
        for row in handoff.boundaries
    ]


def _run_die_cts(
    handoff: OptHandoff,
    die: str,
    input_odb: Path,
    sdc_text: str,
    platform: Path,
    output: Path,
    openroad: str,
    buffer: str,
    cluster_size: int,
    cluster_diameter: float,
    delay_buffer_derate: float,
    clock: HBTBoundary,
    load_cell: Optional[str] = None,
    load_liberty: Optional[Path] = None,
    load_lef: Optional[Path] = None,
    clock_period_ns: Optional[float] = None,
    clock_source_latency_ns: float = 0.0,
    signal_layer: Optional[str] = None,
    clock_layer: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    output.mkdir(parents=True, exist_ok=True)
    sdc = output / "input.sdc"
    write_sdc_projection(sdc, project_sdc_for_handoff(
        handoff, die, sdc_text, source_label="CTS input ({})".format(die),
        platform=platform, openroad=openroad, evidence_dir=output / "sdc_evidence",
    ))
    _lefs, libs = _platform_files(platform, die)
    plan = output / "clock_load.tsv"
    if load_cell:
        plan.write_text(
            "{}\t{}\t{}\t{}\t{}\n".format(
                load_cell + "_I",
                load_cell,
                clock.physical_net or clock.net_name,
                clock.x_dbu,
                clock.y_dbu,
            ),
            encoding="utf-8",
        )
    else:
        plan.write_text("", encoding="utf-8")
    handoff_ports = clock.source_port if die == clock.source_die else clock.sink_port
    boundary_ports = output / "boundary_ports.txt"
    boundary_ports.write_text(
        "".join(row.role_on(die)[1] + "\n" for row in handoff.boundaries),
        encoding="utf-8",
    )
    common_warnings = [
        (
            "CTS-0041",
            r'Net ".+" has 1 sinks\. Skipping\.\.\.',
            "single-sink-clock-needs-no-tree",
        ),
        (
            "CTS-0041",
            r'Net ".+" has 0 sinks\. Skipping\.\.\.',
            "boundary-only-clock-needs-no-local-tree",
        ),
        (
            "CTS-0041",
            r"message limit \(1000\) reached\. This message will no longer print\.",
            "single-sink-clock-message-limit",
        ),
        (
            "RSZ-0062",
            r"Unable to repair all setup violations\.",
            "setup-not-closed",
        ),
    ]
    _run_openroad(
        DIE_CTS_TCL,
        {
            "CTS_DIE": die,
            "CTS_INPUT_ODB": str(input_odb),
            "CTS_DIE_LIBS": "\n".join(str(path) for path in libs),
            "CTS_DIE_SDC": str(sdc),
            "CTS_CLOCK_NET": clock.physical_net or clock.net_name,
            "CTS_SET_RC": str(platform / "setRC.tcl"),
            "CTS_SIGNAL_LAYER": signal_layer or "metal3",
            "CTS_CLOCK_LAYER": clock_layer or "metal5",
            "CTS_BUFFER": buffer,
            "CTS_CLUSTER_SIZE": str(cluster_size),
            "CTS_CLUSTER_DIAMETER": str(cluster_diameter),
            "CTS_DELAY_BUFFER_DERATE": str(delay_buffer_derate),
            "CTS_CLOCK_LOAD_PLAN": str(plan) if load_cell else "",
            "CTS_CLOCK_LOAD_LIB": str(load_liberty) if load_liberty else "",
            "CTS_CLOCK_LOAD_LEF": str(load_lef) if load_lef else "",
            "CTS_EXPECTED_CLOCK_LOADS": "1" if load_cell else "0",
            "CTS_ALIGN_X": str(clock.x_dbu),
            "CTS_ALIGN_Y": str(clock.y_dbu),
            "CTS_HANDOFF_PORTS": handoff_ports,
            # Keep unbounded design data out of the execve environment.  A
            # max-cut partition can expose tens of thousands of boundary
            # ports, which exceeds Linux's per-string/environment limit.
            "CTS_BOUNDARY_PORTS_FILE": str(boundary_ports),
            "CTS_BOUNDARY_NETS": str(output / "boundary_nets.tsv"),
            "CTS_SINKS_TSV": str(output / "clock_sinks.tsv"),
            "CTS_HANDOFF_TSV": str(output / "clock_handoff.tsv"),
            "CTS_METRICS_TSV": str(output / "metrics.tsv"),
            "CTS_CHECK_SETUP_REPORT": str(output / "check_setup.rpt"),
            "CTS_SETUP_REPORT": str(output / "setup.rpt"),
            "CTS_HOLD_REPORT": str(output / "hold.rpt"),
            "CTS_OUTPUT_ODB": str(output / "design.odb"),
            "CTS_OUTPUT_DEF": str(output / "design.def"),
            "CTS_OUTPUT_VERILOG": str(output / "design.v"),
            "CTS_OUTPUT_SDC": str(output / "design.sdc"),
        },
        output / "cts.log",
        openroad,
        common_warnings,
    )
    rows = _boundary_rows(handoff)
    physical_by_port = _read_boundary_net_map(output / "boundary_nets.tsv")
    if len(physical_by_port) != len(handoff.boundaries):
        raise ThreeDCTSError(
            "{} exported {} HBT boundary nets; expected {}".format(
                die, len(physical_by_port), len(handoff.boundaries)
            )
        )
    _repair_boundary_verilog(output / "design.v", die, rows, physical_by_port)
    _validate_boundary_verilog(output / "design.v", die, rows)
    if die == clock.sink_die:
        if clock_period_ns is None:
            raise ThreeDCTSError("receiving-die CTS requires the master period")
        handoff_values = {}
        latency = _sink_latency(
            output / "clock_sinks.tsv",
            clock_source_latency_ns,
            clock_source_latency_ns + clock_period_ns / 2.0,
        )
    else:
        handoff_values = _handoff_values(
            output / "clock_handoff.tsv", clock.source_port
        )
        latency = _sink_latency(output / "clock_sinks.tsv")
    return _read_key_values(output / "metrics.tsv"), latency, handoff_values


POST_CTS_HOLD_REPAIR_TCL = PORT_INVENTORY_TCL + PG_CONNECT_TCL + r"""
read_db $::env(CTS_REPAIR_INPUT_ODB)
foreach lib [split $::env(CTS_REPAIR_LIBS) "\n"] {
  if {$lib ne ""} { read_liberty $lib }
}
open3d_validate_sdc_ports $::env(CTS_REPAIR_SDC).ports.tsv
set_cmd_units -time ns
read_sdc $::env(CTS_REPAIR_SDC)
source $::env(CTS_REPAIR_SET_RC)
set_wire_rc -signal -layer $::env(CTS_REPAIR_SIGNAL_LAYER)
set_wire_rc -clock -layer $::env(CTS_REPAIR_CLOCK_LAYER)
# The CTS-produced SDC already marks the real clock propagated.  Do not apply
# propagation to the virtual boundary reference appended for data budgets.

foreach pattern {*/TAPCELL* */FILLCELL* */HBT_*} {
  set cells [get_lib_cells -quiet $pattern]
  if {[llength $cells] > 0} { set_dont_use $cells }
}

set block [ord::get_db_block]
set pre_instances [llength [$block getInsts]]
set physical_hbt_cells [get_cells -quiet {HBT_*}]
if {[llength $physical_hbt_cells] != $::env(CTS_REPAIR_EXPECTED_HBTS)} {
  error "$::env(CTS_REPAIR_DIE) physical HBT count is not preserved"
}
set_dont_touch $physical_hbt_cells

# Post-CTS repair is data-path only.  Preserve every synthesized clock cell and
# every net recognized by OpenSTA as part of the propagated clock network.
set protected_clock_cells 0
set protected_clock_cell_names [dict create]
foreach inst [$block getInsts] {
  set name [$inst getName]
  if {[string match "clkbuf_*" $name]} {
    set cell [get_cells -quiet [list $name]]
    if {[llength $cell] == 1} {
      set_dont_touch $cell
      dict set protected_clock_cell_names $name 1
      incr protected_clock_cells
    }
  }
}
set protected_clock_nets 0
foreach db_net [sta::find_all_clk_nets] {
  set net [get_nets -quiet [list [$db_net getName]]]
  if {[llength $net] == 1} {
    set_dont_touch $net
    incr protected_clock_nets
  }
}
if {$protected_clock_cells == 0 || $protected_clock_nets == 0} {
  error "$::env(CTS_REPAIR_DIE) post-CTS repair found no protected clock tree"
}

set protected_boundary_nets 0
foreach db_net [$block getNets] {
  if {[llength [$db_net getBTerms]] <= 1} { continue }
  set net [get_nets -quiet [list [$db_net getName]]]
  if {[llength $net] == 1} {
    set_dont_touch $net
    incr protected_boundary_nets
  }
}

estimate_parasitics -placement
set pre_wns_hold [sta::time_sta_ui [sta::worst_slack_cmd min]]
set pre_tns_hold [sta::time_sta_ui [sta::total_negative_slack_cmd min]]

set hold_buffer_limit_reached 0
if {[catch {
  repair_timing -hold \
    -max_passes 1 \
    -max_buffer_percent $::env(CTS_REPAIR_MAX_BUFFER_PERCENT) \
    -max_utilization $::env(CTS_REPAIR_MAX_UTILIZATION)
} hold_error]} {
  if {$hold_error eq "RSZ-0060" || [string match {*Max buffer count reached*} $hold_error]} {
    set hold_buffer_limit_reached 1
    puts "OPEN3D_POST_CTS_HOLD_BOUNDED die=$::env(CTS_REPAIR_DIE) reason=max_buffer_count"
  } else {
    error $hold_error
  }
}
detailed_placement -max_displacement {500 500}
check_placement -verbose
estimate_parasitics -placement

lassign [open3d_connect_pg_iterms 1] pg_iterms pg_iterms_repaired floating_pg_iterms
if {$floating_pg_iterms != 0} {
  error "$::env(CTS_REPAIR_DIE) post-CTS repair left floating PG ITerms"
}
set missing_clock_cells 0
foreach name [dict keys $protected_clock_cell_names] {
  if {[$block findInst $name] eq "NULL"} {
    puts "OPEN3D_POST_CTS_MISSING_CLOCK_CELL die=$::env(CTS_REPAIR_DIE) cell=$name"
    incr missing_clock_cells
  }
}
if {$missing_clock_cells != 0} {
  error "$::env(CTS_REPAIR_DIE) post-CTS repair removed $missing_clock_cells protected clock cells"
}

set ports [open $::env(CTS_REPAIR_EXPECTED_PORTS) r]
set port_count 0
while {[gets $ports port] >= 0} {
  if {$port eq ""} { continue }
  if {[$block findBTerm $port] eq "NULL"} { error "lost HBT BTerm $port" }
  incr port_count
}
close $ports
if {$port_count != $::env(CTS_REPAIR_EXPECTED_HBTS)} {
  error "$::env(CTS_REPAIR_DIE) post-CTS repair lost HBT ports"
}

set metrics [open $::env(CTS_REPAIR_METRICS) w]
puts $metrics "pre_instances\t$pre_instances"
puts $metrics "post_instances\t[llength [$block getInsts]]"
puts $metrics "inserted_instances\t[expr {[llength [$block getInsts]] - $pre_instances}]"
puts $metrics "protected_clock_cells\t$protected_clock_cells"
puts $metrics "protected_clock_nets\t$protected_clock_nets"
puts $metrics "protected_boundary_nets\t$protected_boundary_nets"
puts $metrics "hold_buffer_limit_reached\t$hold_buffer_limit_reached"
puts $metrics "pre_wns_hold\t$pre_wns_hold"
puts $metrics "pre_tns_hold\t$pre_tns_hold"
puts $metrics "post_wns_hold\t[sta::time_sta_ui [sta::worst_slack_cmd min]]"
puts $metrics "post_tns_hold\t[sta::time_sta_ui [sta::total_negative_slack_cmd min]]"
puts $metrics "pg_iterms\t$pg_iterms"
puts $metrics "pg_iterms_repaired\t$pg_iterms_repaired"
puts $metrics "floating_pg_iterms\t$floating_pg_iterms"
close $metrics

set boundary_map [open $::env(CTS_REPAIR_BOUNDARY_NETS) w]
set ports [open $::env(CTS_REPAIR_EXPECTED_PORTS) r]
while {[gets $ports port] >= 0} {
  if {$port eq ""} { continue }
  set bterm [$block findBTerm $port]
  if {$bterm eq "NULL" || [$bterm getNet] eq "NULL"} {
    error "cannot export post-CTS HBT boundary net for $port"
  }
  puts $boundary_map "$port\t[[$bterm getNet] getName]"
}
close $ports
close $boundary_map

report_checks -path_delay max -group_path_count 20 > $::env(CTS_REPAIR_SETUP_REPORT)
report_checks -path_delay min -group_path_count 20 > $::env(CTS_REPAIR_HOLD_REPORT)
write_db $::env(CTS_REPAIR_OUTPUT_ODB)
write_def $::env(CTS_REPAIR_OUTPUT_DEF)
write_verilog $::env(CTS_REPAIR_OUTPUT_VERILOG)
puts "OPEN3D_POST_CTS_HOLD_REPAIR_DONE die=$::env(CTS_REPAIR_DIE)"
"""


def _post_cts_hold_repair_die(
    handoff: OptHandoff,
    die: str,
    input_odb: Path,
    base_sdc: Path,
    snapshot: Mapping[str, object],
    platform: Path,
    output: Path,
    openroad: str,
    signal_layer: str,
    clock_layer: str,
    max_buffer_percent: float = 1.0,
    max_utilization: float = 75.0,
) -> Dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    _, libs = _platform_files(platform, die)
    repair_sdc = output / "repair.sdc"
    boundary_counts = _post_cts_boundary_sdc(
        die, base_sdc, snapshot, handoff, repair_sdc
    )
    write_sdc_projection(repair_sdc, project_sdc_for_handoff(
        handoff, die, repair_sdc.read_text(encoding="utf-8"), source_label=str(base_sdc),
        def_paths={name: input_odb.with_suffix(".def") if name == die else item.deffile
                   for name, item in handoff.dies.items()},
        odb_paths={name: input_odb if name == die else item.odb
                   for name, item in handoff.dies.items()},
        platform=platform, openroad=openroad, evidence_dir=output / "sdc_evidence",
    ))
    expected_ports = output / "expected_hbt_ports.txt"
    expected_ports.write_text(
        "".join(boundary.role_on(die)[1] + "\n" for boundary in handoff.boundaries),
        encoding="utf-8",
    )
    metrics_path = output / "metrics.tsv"
    _run_openroad(
        POST_CTS_HOLD_REPAIR_TCL,
        {
            "CTS_REPAIR_DIE": die,
            "CTS_REPAIR_INPUT_ODB": str(input_odb),
            "CTS_REPAIR_LIBS": "\n".join(str(path) for path in libs),
            "CTS_REPAIR_SDC": str(repair_sdc),
            "CTS_REPAIR_SET_RC": str(platform / "setRC.tcl"),
            "CTS_REPAIR_SIGNAL_LAYER": signal_layer,
            "CTS_REPAIR_CLOCK_LAYER": clock_layer,
            "CTS_REPAIR_EXPECTED_HBTS": str(len(handoff.boundaries)),
            "CTS_REPAIR_EXPECTED_PORTS": str(expected_ports),
            "CTS_REPAIR_MAX_BUFFER_PERCENT": str(max_buffer_percent),
            "CTS_REPAIR_MAX_UTILIZATION": str(max_utilization),
            "CTS_REPAIR_METRICS": str(metrics_path),
            "CTS_REPAIR_BOUNDARY_NETS": str(output / "boundary_nets.tsv"),
            "CTS_REPAIR_SETUP_REPORT": str(output / "setup.rpt"),
            "CTS_REPAIR_HOLD_REPORT": str(output / "hold.rpt"),
            "CTS_REPAIR_OUTPUT_ODB": str(output / "design.odb"),
            "CTS_REPAIR_OUTPUT_DEF": str(output / "design.def"),
            "CTS_REPAIR_OUTPUT_VERILOG": str(output / "design.v"),
        },
        output / "repair.log",
        openroad,
        [
            ("RSZ-0064", r"Unable to repair all hold checks within margin\.", "hold-margin-not-closed"),
            ("RSZ-0066", r"Unable to repair all hold violations\.", "hold-not-closed"),
            *RESIZER_BUFFERING_WARNINGS,
            (
                "RSZ-3018",
                r"insertBufferBeforeLoads: skip net .+ because at least one load is dont-touch",
                "protected-clock-or-hbt-load",
            ),
        ],
    )
    rows = _boundary_rows(handoff)
    physical_by_port = _read_boundary_net_map(output / "boundary_nets.tsv")
    if len(physical_by_port) != len(handoff.boundaries):
        raise ThreeDCTSError("post-CTS repair did not export every HBT boundary")
    _repair_boundary_verilog(output / "design.v", die, rows, physical_by_port)
    _validate_boundary_verilog(output / "design.v", die, rows)
    shutil.copy2(base_sdc, output / "design.sdc")
    return {
        "boundary_constraints": boundary_counts,
        "limits": {
            "max_buffer_percent": max_buffer_percent,
            "max_utilization_percent": max_utilization,
        },
        "timing": _read_key_values(metrics_path),
    }


CTS_RC_CONSTRUCTION_TCL = r"""
read_lef $::env(CTS_TECH_LEF)
foreach lef [split $::env(CTS_DIE_LEFS) "\n"] {
  if {$lef ne ""} { read_lef $lef }
}
read_lef $::env(CTS_HBT_LEF)
foreach lib [split $::env(CTS_LIBS) "\n"] {
  if {$lib ne ""} { read_liberty $lib }
}
read_liberty $::env(CTS_HBT_LIB)
read_verilog $::env(CTS_ASSEMBLY_V)
link_design top_3d
set block [ord::get_db_block]

set placement [open $::env(CTS_PLACEMENT_TSV) r]
set missing 0
while {[gets $placement line] >= 0} {
  if {$line eq ""} { continue }
  set fields [split $line "\t"]
  set kind [lindex $fields 0]
  set die [lindex $fields 1]
  set name [lindex $fields 2]
  if {$kind eq "I"} {
    set escaped [string map [list "/" "\\/"] $name]
    set inst [$block findInst "u_${die}/$escaped"]
    # CTS-generated instance names can already contain OpenDB's escaped slash
    # spelling (for example ``foo\\/bar``).  Re-escaping that spelling hides a
    # real timed instance, so retry the exact database name before declaring a
    # placement miss.
    if {$inst eq "NULL"} {
      set inst [$block findInst "u_${die}/$name"]
    }
    if {$inst eq "NULL"} {
      if {[regexp {^(TAPCELL|FILLCELL|HBT_(BOTIN|TOPIN))} [lindex $fields 3]]} { continue }
      incr missing
      puts "CTS_MISSING_ASSEMBLY_INSTANCE die=$die name=$name"
      continue
    }
    $inst setLocation [lindex $fields 4] [lindex $fields 5]
    $inst setOrient [lindex $fields 6]
    $inst setPlacementStatus PLACED
  } elseif {$kind eq "P" && $die eq "bottom"} {
    set bterm [$block findBTerm $name]
    if {$bterm eq "NULL"} {
      if {[string match HBT_* $name] || $name in {VDD VSS}} { continue }
      error "bottom physical port $name is missing from assembly"
    }
    set layer [[ord::get_db_tech] findLayer [lindex $fields 3]]
    set bpin [odb::dbBPin_create $bterm]
    odb::dbBox_create $bpin $layer [lindex $fields 4] [lindex $fields 5] \
      [lindex $fields 6] [lindex $fields 7]
    $bpin setPlacementStatus PLACED
  }
}
close $placement
if {$missing != 0} { error "assembly misses $missing timed physical instances" }

set plan [open $::env(CTS_BOUNDARY_PLAN) r]
while {[gets $plan line] >= 0} {
  if {$line eq ""} { continue }
  set fields [split $line "\t"]
  set inst [$block findInst [lindex $fields 1]]
  if {$inst eq "NULL"} { error "missing timed HBT [lindex $fields 1]" }
  $inst setLocation [lindex $fields 5] [lindex $fields 6]
  $inst setPlacementStatus PLACED
}
close $plan

source $::env(CTS_SET_RC)
set_cmd_units -time ns
read_sdc $::env(CTS_ASSEMBLY_SDC)
set masters [all_clocks]
if {[llength $masters] == 0} { error "clock HBT assembly has no master clocks" }
set plan [open $::env(CTS_BOUNDARY_PLAN) r]
set formal_clock_hbts 0
while {[gets $plan line] >= 0} {
  if {$line eq ""} { continue }
  set fields [split $line "\t"]
  if {[lindex $fields 4] ne "1"} { continue }
  set source [get_pins -quiet [lindex $fields 1]/[lindex $fields 2]]
  set sink [get_pins -quiet [lindex $fields 1]/[lindex $fields 3]]
  if {[llength $source] != 1 || [llength $sink] != 1} {
    error "cannot resolve formal clock HBT pins"
  }
  incr formal_clock_hbts
}
close $plan
set_propagated_clock [all_clocks]
# Reading the two CTS Verilog files reconstructs logical nets but loses the
# dbSigType::CLOCK annotation stored in each die ODB.  OpenROAD's placement
# estimator selects clock RC only from that annotation, so restore it from the
# linked STA graph before estimating any post-CTS wire parasitics.
set clock_net_count 0
foreach net [sta::find_all_clk_nets] {
  $net setSigType CLOCK
  incr clock_net_count
}
if {$clock_net_count == 0} { error "assembly has no propagated clock nets" }
estimate_parasitics -placement -spef_file $::env(CTS_ESTIMATED_SPEF)
puts "OPEN3D_CTS_RC_CONSTRUCTION formal_clock_hbts=$formal_clock_hbts clock_nets=$clock_net_count"
"""


PASSIVE_CTS_STA_TCL = STA_COMPLETENESS_TCL + r"""
if {[info exists ::env(OPEN3D_LIBERTY_SUPPRESSIONS)]
    && [file exists $::env(OPEN3D_LIBERTY_SUPPRESSIONS)]} {
  source $::env(OPEN3D_LIBERTY_SUPPRESSIONS)
}
read_lef $::env(CTS_TECH_LEF)
foreach lef [split $::env(CTS_DIE_LEFS) "\n"] {
  if {$lef ne ""} { read_lef $lef }
}
foreach lib [split $::env(CTS_LIBS) "\n"] {
  if {$lib ne ""} { read_liberty $lib }
}
read_verilog $::env(CTS_PASSIVE_V)
link_design top_3d
set_cmd_units -time ns
read_sdc $::env(CTS_PASSIVE_SDC)
read_spef $::env(CTS_PASSIVE_SPEF)
set_propagated_clock [all_clocks]
check_setup -verbose > $::env(CTS_CHECK_SETUP_REPORT)
open3d_check_sta_completeness passive_global_post_cts 1
report_checks -path_delay max -group_path_count 20 > $::env(CTS_SETUP_REPORT)
report_checks -path_delay min -group_path_count 20 > $::env(CTS_HOLD_REPORT)
report_parasitic_annotation -report_unannotated > $::env(CTS_ANNOTATION_REPORT)

proc open3d_passive_prop {object property} {
  if {[catch {set value [get_property $object $property]}]} { return "" }
  return $value
}
set sinks [open $::env(CTS_SINKS_TSV) w]
# A register clock pin is timed on either the rising or falling edge.  Dump
# only pins attached to the formal primary clock and retain that active edge.
# Iterating all clock pins used to include unclocked latch gates whose missing
# transition properties read back as zero and inflated reported skew by an
# entire clock phase.
foreach {active_edge active_pins} [list \
    rise [all_registers -clock_pins -rise_clock $::env(CTS_PRIMARY_CLOCK_NAME)] \
    fall [all_registers -clock_pins -fall_clock $::env(CTS_PRIMARY_CLOCK_NAME)]] {
  foreach pin $active_pins {
    set name [get_full_name $pin]
    set die [expr {[string match "u_bottom/*" $name] ? "bottom" : \
      ([string match "u_top/*" $name] ? "top" : "unknown")}]
    puts $sinks [join [list $die $name $active_edge \
      [open3d_passive_prop $pin arrival_max_rise] [open3d_passive_prop $pin arrival_max_fall] \
      [open3d_passive_prop $pin arrival_min_rise] [open3d_passive_prop $pin arrival_min_fall]] "\t"]
  }
}
close $sinks

# Keep the primary-clock table above unchanged for the passive handoff.  The
# sidecar also reports local clock domains, without mixing waveform phases,
# active edges or min/max analyses.  Arrival properties are not clock-filtered:
# shared-domain pins must therefore remain explicitly ambiguous.
set pins_by_clock [dict create]
set clocks_by_pin [dict create]
foreach clock [all_clocks] {
  set clock_name [get_name $clock]
  set clock_pins [all_registers -clock $clock -clock_pins]
  dict set pins_by_clock $clock_name $clock_pins
  foreach pin $clock_pins {
    dict lappend clocks_by_pin [get_full_name $pin] $clock_name
  }
}
set domain_sinks [open $::env(CTS_SINKS_BY_CLOCK_TSV) w]
puts $domain_sinks "clock\tdie\tpin\tactive_edge\tarrival_max_rise_ns\tarrival_max_fall_ns\tarrival_min_rise_ns\tarrival_min_fall_ns\tclock_domain_count"
dict for {clock_name clock_pins} $pins_by_clock {
  if {[llength $clock_pins] == 0} {
    puts $domain_sinks [join [list $clock_name none "" "" "" "" "" "" 0] "\t"]
  }
  foreach {active_edge active_pins} [list \
      rise [all_registers -clock_pins -rise_clock $clock_name] \
      fall [all_registers -clock_pins -fall_clock $clock_name]] {
    foreach pin $active_pins {
      set name [get_full_name $pin]
      set die [expr {[string match "u_bottom/*" $name] ? "bottom" : \
        ([string match "u_top/*" $name] ? "top" : "unknown")}]
      set domains [llength [lsort -unique [dict get $clocks_by_pin $name]]]
      puts $domain_sinks [join [list $clock_name $die $name $active_edge \
        [open3d_passive_prop $pin arrival_max_rise] [open3d_passive_prop $pin arrival_max_fall] \
        [open3d_passive_prop $pin arrival_min_rise] [open3d_passive_prop $pin arrival_min_fall] \
        $domains] "\t"]
    }
  }
}
close $domain_sinks

# Export the one formal clock-HBT interface directly from the same passive RC
# network used for the authoritative global timing report.  Both max/min and
# rise/fall are required: source CTS consumes the downstream load and insertion
# range, while receiving CTS consumes the far-side arrival and slew.
set interface [open $::env(CTS_CLOCK_INTERFACE_TSV) w]
puts $interface "hbt_id\tpoint\tanalysis\ttransition\tavailable\tarrival_ns\tslew_ns\tdownstream_cap_pf\tpath_resistance_ui\telmore_ns"
set plan [open $::env(CTS_PASSIVE_PLAN) r]
set clock_hbt_count 0
set interface_rows 0
while {[gets $plan line] >= 0} {
  if {$line eq ""} { continue }
  lassign [split $line "\t"] hbt_id net_name driver_kind driver_name near_node_id far_node_id
  if {$hbt_id ne $::env(CTS_CLOCK_HBT_ID)} { continue }
  incr clock_hbt_count
  set net [get_nets -quiet [list $net_name]]
  if {[llength $net] != 1} { error "cannot resolve passive clock-HBT net $net_name" }
  set driver [sta::net_driver_pins $net]
  if {[llength $driver] != 1} {
    error "passive clock-HBT net $net_name has [llength $driver] logical drivers"
  }
  foreach {point node_id} [list near $near_node_id far $far_node_id] {
    foreach analysis {max min} {
      foreach transition {rise fall} {
        set timing [get_parasitic_cutpoint_timing -$transition -$analysis \
          -clock $::env(CTS_PRIMARY_CLOCK_NAME) $driver $node_id]
        if {[dict size $timing] == 0} {
          puts $interface "$hbt_id\t$point\t$analysis\t$transition\t0\t\t\t\t\t"
          continue
        }
        puts $interface [join [list \
          $hbt_id $point $analysis $transition 1 \
          [sta::time_sta_ui [dict get $timing arrival]] \
          [sta::time_sta_ui [dict get $timing slew]] \
          [expr {[dict get $timing downstream_cap] * 1.0e12}] \
          [sta::resistance_sta_ui [dict get $timing path_resistance]] \
          [sta::time_sta_ui [dict get $timing elmore]]] "\t"]
        incr interface_rows
      }
    }
  }
}
close $plan
close $interface
if {$clock_hbt_count != 1 || $interface_rows != 8} {
  error "passive clock-HBT coverage hbt=$clock_hbt_count rows=$interface_rows; expected 1 and 8"
}

set boundary_rows 0
if {$::env(CTS_QUERY_ALL_BOUNDARIES)} {
  set boundary_cutpoints [open $::env(CTS_BOUNDARY_CUTPOINTS_TSV) w]
  puts $boundary_cutpoints "hbt_id\tpoint\tanalysis\ttransition\tavailable\tarrival_ns\tslew_ns\trequired_ns\tslack_ns\twire_delay_ns\telmore_ns\tdownstream_cap_ui\tpath_resistance_ui\tdownstream_load_count\tconstrained_load_count\tconstrained"
  set boundary_loads [open $::env(CTS_BOUNDARY_LOADS_TSV) w]
  set plan [open $::env(CTS_PASSIVE_PLAN) r]
  set boundary_count 0
  while {[gets $plan line] >= 0} {
    if {$line eq ""} { continue }
    lassign [split $line "\t"] hbt_id net_name driver_kind driver_name near_node_id far_node_id
    set net [get_nets -quiet [list $net_name]]
    if {[llength $net] != 1} { error "cannot resolve passive-HBT net $net_name" }
    set driver [sta::net_driver_pins $net]
    if {[llength $driver] != 1} {
      error "passive-HBT net $net_name has [llength $driver] logical drivers"
    }
    set wrote_load 0
    foreach {point node_id} [list near $near_node_id far $far_node_id] {
      foreach analysis {max min} {
        foreach transition {rise fall} {
          set timing [get_parasitic_cutpoint_timing -$transition -$analysis $driver $node_id]
          if {[dict size $timing] == 0} {
            puts $boundary_cutpoints [join [list \
              $hbt_id $point $analysis $transition 0 \
              "" "" "" "" "" "" "" "" 0 0 0] "\t"]
            continue
          }
          set constrained [expr {[dict get $timing constrained] ? 1 : 0}]
          puts $boundary_cutpoints [join [list \
            $hbt_id $point $analysis $transition 1 \
            [sta::time_sta_ui [dict get $timing arrival]] \
            [sta::time_sta_ui [dict get $timing slew]] \
            [sta::time_sta_ui [dict get $timing required]] \
            [sta::time_sta_ui [dict get $timing slack]] \
            [sta::time_sta_ui [dict get $timing wire_delay]] \
            [sta::time_sta_ui [dict get $timing elmore]] \
            [sta::capacitance_sta_ui [dict get $timing downstream_cap]] \
            [sta::resistance_sta_ui [dict get $timing path_resistance]] \
            [dict get $timing downstream_load_count] \
            [dict get $timing constrained_load_count] \
            $constrained] "\t"]
          incr boundary_rows
          if {$point eq "near" && $analysis eq "max" && $transition eq "rise"} {
            puts $boundary_loads "$hbt_id\t[sta::capacitance_sta_ui [dict get $timing downstream_cap]]"
            set wrote_load 1
          }
        }
      }
    }
    if {!$wrote_load} {
      # Constants and logically inactive crossings have no STA timing state.
      # Keep their coverage explicit but do not invent a repair load.
      puts $boundary_loads "$hbt_id\t0"
    }
    incr boundary_count
  }
  close $plan
  close $boundary_loads
  close $boundary_cutpoints
  if {$boundary_count != $::env(CTS_EXPECTED_HBTS)} {
    error "passive boundary coverage hbts=$boundary_count expected=$::env(CTS_EXPECTED_HBTS)"
  }
}
set metrics [open $::env(CTS_METRICS_TSV) w]
puts $metrics "wns_setup\t[sta::time_sta_ui [sta::worst_slack_cmd max]]"
puts $metrics "tns_setup\t[sta::time_sta_ui [sta::total_negative_slack_cmd max]]"
puts $metrics "wns_hold\t[sta::time_sta_ui [sta::worst_slack_cmd min]]"
puts $metrics "tns_hold\t[sta::time_sta_ui [sta::total_negative_slack_cmd min]]"
puts $metrics "clock_count\t[llength [all_clocks]]"
puts $metrics "generated_clock_hbt_count\t0"
puts $metrics "formal_clock_hbt_count\t1"
puts $metrics "passive_hbt_rc\t1"
puts $metrics "clock_hbt_cutpoint_rows\t$interface_rows"
puts $metrics "boundary_cutpoint_rows\t$boundary_rows"
close $metrics
puts "OPEN3D_PASSIVE_GLOBAL_CTS_STA"
"""


def _assembly_skew(path: Path) -> Dict[str, object]:
    by_die: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        die: {
            analysis: {"rise": [], "fall": []}
            for analysis in ("max", "min")
        }
        for die in ("bottom", "top")
    }
    count = 0
    sink_names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if (
            len(fields) != 7
            or fields[0] not in by_die
            or fields[2] not in {"rise", "fall"}
        ):
            raise ThreeDCTSError("invalid assembly clock-sink row")
        die, active_edge = fields[0], fields[2]
        edge_offset = 0 if active_edge == "rise" else 1
        by_die[die]["max"][active_edge].extend(
            _finite_values((fields[3 + edge_offset],))
        )
        by_die[die]["min"][active_edge].extend(
            _finite_values((fields[5 + edge_offset],))
        )
        count += 1
        sink_names.add(fields[1])
    if count == 0:
        raise ThreeDCTSError("assembly has no primary-clock sinks")
    metrics: Dict[str, object] = {"sink_count": count, "dies": {}}
    all_edges = {
        analysis: {"rise": [], "fall": []}
        for analysis in ("max", "min")
    }
    for die, analyses in by_die.items():
        die_metrics = {}
        for edge in ("rise", "fall"):
            candidates = []
            for analysis in ("max", "min"):
                values = analyses[analysis][edge]
                all_edges[analysis][edge].extend(values)
                if values:
                    candidates.append(
                        {
                            "analysis": analysis,
                            "earliest_ns": min(values),
                            "latest_ns": max(values),
                            "skew_ns": max(values) - min(values),
                        }
                    )
            if candidates:
                die_metrics[edge] = max(
                    candidates, key=lambda candidate: candidate["skew_ns"]
                )
        if not die_metrics:
            raise ThreeDCTSError("assembly has no finite {} clock sinks".format(die))
        metrics["dies"][die] = die_metrics
    analysis_edge_skews = {}
    for analysis, edges in all_edges.items():
        analysis_edge_skews[analysis] = {
            edge: max(values) - min(values)
            for edge, values in edges.items()
            if values
        }
    finite_skews = [
        skew
        for edges in analysis_edge_skews.values()
        for skew in edges.values()
    ]
    if not finite_skews:
        raise ThreeDCTSError("assembly has no finite primary-clock arrivals")
    metrics["global_skew_ns"] = max(finite_skews)
    metrics["analysis_edge_skew_ns"] = analysis_edge_skews
    metrics["global_skew_role"] = "primary-clock-active-edge-analysis-separated"
    metrics["global_skew_valid_as_single_clock"] = None
    metrics["per_clock_report_status"] = "unavailable"
    metrics["per_clock"] = {}
    domain_path = path.with_name("clock_sinks_by_clock.tsv")
    if domain_path.is_file():
        metrics.update(_per_clock_skew(domain_path, sink_names))
    return metrics


_CLOCK_SINK_COLUMNS = (
    "clock", "die", "pin", "active_edge", "arrival_max_rise_ns", "arrival_max_fall_ns",
    "arrival_min_rise_ns", "arrival_min_fall_ns", "clock_domain_count",
)


def _per_clock_skew(path: Path, expected_sinks: Set[str]) -> Dict[str, object]:
    """Reduce each clock/active-edge/analysis separately; reject ambiguous pins.

    ``expected_sinks`` is the primary-clock table, not all domain sinks.  Local
    auxiliary-clock pins are expected additions, not a coverage error.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != _CLOCK_SINK_COLUMNS:
        raise ThreeDCTSError("invalid per-clock sink-table header")
    groups: Dict[str, list] = {}
    pin_domains: Dict[str, Set[str]] = {}
    seen = set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(_CLOCK_SINK_COLUMNS) or not fields[0]:
            raise ThreeDCTSError("invalid per-clock sink row")
        clock, die, pin, active_edge = fields[:4]
        try:
            domain_count = int(fields[8])
        except ValueError as exc:
            raise ThreeDCTSError("invalid clock-domain count") from exc
        rows = groups.setdefault(clock, [])
        if die == "none" and not pin and domain_count == 0 and not any(fields[3:8]):
            continue
        if (die not in {"bottom", "top"} or not pin or domain_count < 1
                or active_edge not in {"rise", "fall"}):
            raise ThreeDCTSError("invalid per-clock sink identity")
        domains = pin_domains.setdefault(pin, set())
        identity = (clock, pin, active_edge)
        if identity in seen:
            raise ThreeDCTSError("duplicate clock/sink row: {}/{}".format(clock, pin))
        seen.add(identity)
        domains.add(clock)
        rows.append((die, pin, active_edge, fields[4:8], domain_count))

    per_clock: Dict[str, object] = {}
    for clock, rows in groups.items():
        by_die = {}
        ambiguous = set()
        nonfinite = set()
        for die, pin, active_edge, raw, domain_count in rows:
            if domain_count > 1 or len(pin_domains[pin]) > 1:
                ambiguous.add(pin)
            edge_offset = 0 if active_edge == "rise" else 1
            values = _finite_values((raw[edge_offset], raw[2 + edge_offset]))
            if len(values) != 2:
                nonfinite.add(pin)
                continue
            analyses = by_die.setdefault(die, {
                analysis: {"rise": [], "fall": []} for analysis in ("max", "min")
            })
            analyses["max"][active_edge].append(values[0])
            analyses["min"][active_edge].append(values[1])
        status = (
            "no-sinks" if not rows else
            "ambiguous-shared-clock-pins" if ambiguous else
            "incomplete-arrivals" if nonfinite else "valid"
        )
        item = {
            "status": status,
            "sink_count": len({row[1] for row in rows}),
            "sink_count_by_die": {
                die: len({row[1] for row in rows if row[0] == die})
                for die in ("bottom", "top")
            },
            "ambiguous_sink_count": len(ambiguous),
            "ambiguous_sink_examples": sorted(ambiguous)[:10],
            "nonfinite_sink_count": len(nonfinite),
            "skew_ns": None,
            "analysis_edge_skew_ns": None,
            "dies": {},
        }
        if status == "valid":
            combined = {
                analysis: {"rise": [], "fall": []} for analysis in ("max", "min")
            }
            for die, analyses in by_die.items():
                die_metrics = {}
                for edge in ("rise", "fall"):
                    candidates = []
                    for analysis, edges in analyses.items():
                        values = edges[edge]
                        if not values:
                            continue
                        combined[analysis][edge].extend(values)
                        candidates.append({
                            "analysis": analysis,
                            "earliest_ns": min(values),
                            "latest_ns": max(values),
                            "skew_ns": max(values) - min(values),
                        })
                    if candidates:
                        die_metrics[edge] = max(candidates, key=lambda row: row["skew_ns"])
                item["dies"][die] = die_metrics
            item["analysis_edge_skew_ns"] = {
                analysis: {edge: max(values) - min(values)
                           for edge, values in edges.items() if values}
                for analysis, edges in combined.items()
            }
            item["skew_ns"] = max(
                value for edges in item["analysis_edge_skew_ns"].values()
                for value in edges.values()
            )
        per_clock[clock] = item

    covered = set(pin_domains)
    primary_domains = [
        clock for clock, rows in groups.items()
        if {row[1] for row in rows} == expected_sinks
    ]
    complete = bool(expected_sinks) and len(primary_domains) == 1
    valid = bool(per_clock) and complete and all(
        row["status"] == "valid" for row in per_clock.values()
    )
    return {
        "per_clock_report_status": "valid" if valid else "incomplete-or-ambiguous",
        "per_clock_measurement": "same-clock active-edge sink skew; max/min analyses reduced separately",
        "clock_count": len(per_clock),
        "global_skew_valid_as_single_clock": complete and per_clock[primary_domains[0]]["status"] == "valid",
        "per_clock_sink_coverage": {
            "scope": "primary-clock-table",
            "complete": complete,
            "expected_count": len(expected_sinks),
            "observed_count": len(covered),
            "unassigned_count": len(expected_sinks - covered),
            "auxiliary_clock_sink_count": len(covered - expected_sinks),
        },
        "per_clock": per_clock,
        "max_per_clock_skew_ns": max(
            row["skew_ns"] for row in per_clock.values()
        ) if valid else None,
    }


def _clock_domain_inventory(path: Path) -> Dict[str, Tuple[Tuple[object, ...], ...]]:
    """Read exact clock/sink ownership from the authoritative passive report."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != _CLOCK_SINK_COLUMNS:
        raise ThreeDCTSError("invalid per-clock sink-table header")
    result: Dict[str, Set[Tuple[object, ...]]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(_CLOCK_SINK_COLUMNS) or not fields[0]:
            raise ThreeDCTSError("invalid per-clock sink row")
        clock, die, pin, active_edge = fields[:4]
        try:
            domain_count = int(fields[8])
        except ValueError as exc:
            raise ThreeDCTSError("invalid clock-domain count") from exc
        rows = result.setdefault(clock, set())
        if die == "none" and not pin and domain_count == 0:
            continue
        if (
            die not in {"bottom", "top"}
            or not pin
            or active_edge not in {"rise", "fall"}
            or domain_count < 1
        ):
            raise ThreeDCTSError("invalid clock-domain sink identity")
        identity = (die, pin, active_edge, domain_count)
        if identity in rows:
            raise ThreeDCTSError(
                "duplicate clock-domain sink identity: {}/{}".format(clock, pin)
            )
        rows.add(identity)
    return {
        clock: tuple(sorted(rows))
        for clock, rows in sorted(result.items())
    }


def _skew_target_report(skew: Mapping[str, object], target_ns: float) -> Dict[str, object]:
    groups = skew.get("per_clock", {})
    results = {
        clock: (row["skew_ns"] <= target_ns if row["status"] == "valid" else None)
        for clock, row in groups.items()
    }
    complete = skew.get("per_clock_report_status") == "valid" and bool(results)
    return {
        "scope": "each-independent-clock-domain",
        "target_skew_ns": target_ns,
        "per_clock_target_met": results,
        "target_met": all(results.values()) if complete else None,
        "status": "available" if complete else "unavailable-or-ambiguous",
    }


def _select_coordinated_skew(
    baseline: Mapping[str, object],
    coordinated: Mapping[str, object],
    tolerance_ns: float = 1.0e-6,
    baseline_inventory: Optional[Mapping[str, object]] = None,
    coordinated_inventory: Optional[Mapping[str, object]] = None,
    baseline_clock_buffers: Optional[int] = None,
    coordinated_clock_buffers: Optional[int] = None,
) -> Dict[str, object]:
    """Select a physical tree using only complete same-clock skew evidence.

    Setup/hold values are deliberately absent from this decision.  Before
    comparing the scalar task objective, require identical clock-domain and
    sink inventories so an omitted or reassigned sink can never look like an
    improvement.
    """

    if tolerance_ns < 0.0:
        raise ThreeDCTSError("negative coordinated-skew comparison tolerance")
    if (baseline_inventory is None) != (coordinated_inventory is None):
        raise ThreeDCTSError("incomplete coordinated clock-domain inventory")
    if (
        baseline_inventory is not None
        and baseline_inventory != coordinated_inventory
    ):
        raise ThreeDCTSError(
            "coordinated CTS changed exact clock-domain sink ownership"
        )
    if (
        baseline.get("per_clock_report_status") != "valid"
        or coordinated.get("per_clock_report_status") != "valid"
    ):
        raise ThreeDCTSError(
            "coordinated CTS requires complete, unambiguous per-clock reports"
        )
    baseline_clocks = baseline.get("per_clock")
    coordinated_clocks = coordinated.get("per_clock")
    if not isinstance(baseline_clocks, Mapping) or not isinstance(
        coordinated_clocks, Mapping
    ):
        raise ThreeDCTSError("coordinated CTS has no per-clock inventory")
    if set(baseline_clocks) != set(coordinated_clocks):
        raise ThreeDCTSError("coordinated CTS changed clock-domain ownership")

    inventory = {}
    for clock in sorted(baseline_clocks):
        before = baseline_clocks[clock]
        after = coordinated_clocks[clock]
        if before.get("status") != "valid" or after.get("status") != "valid":
            raise ThreeDCTSError("coordinated CTS has an invalid clock domain")
        before_counts = before.get("sink_count_by_die")
        after_counts = after.get("sink_count_by_die")
        if (
            before.get("sink_count") != after.get("sink_count")
            or before_counts != after_counts
        ):
            raise ThreeDCTSError(
                "coordinated CTS changed sink inventory for {}".format(clock)
            )
        inventory[clock] = {
            "sink_count": before["sink_count"],
            "sink_count_by_die": before_counts,
        }

    baseline_objective = baseline.get("max_per_clock_skew_ns")
    coordinated_objective = coordinated.get("max_per_clock_skew_ns")
    if not isinstance(baseline_objective, (int, float)) or not isinstance(
        coordinated_objective, (int, float)
    ):
        raise ThreeDCTSError("coordinated CTS has no finite skew objective")
    baseline_objective = float(baseline_objective)
    coordinated_objective = float(coordinated_objective)
    if not math.isfinite(baseline_objective) or not math.isfinite(
        coordinated_objective
    ):
        raise ThreeDCTSError("coordinated CTS has a non-finite skew objective")

    if (baseline_clock_buffers is None) != (coordinated_clock_buffers is None):
        raise ThreeDCTSError("incomplete coordinated CTS resource inventory")
    resource_guard_passed = True
    buffer_ratio = None
    if baseline_clock_buffers is not None:
        if (
            type(baseline_clock_buffers) is not int
            or type(coordinated_clock_buffers) is not int
            or baseline_clock_buffers < 0
            or coordinated_clock_buffers < 0
        ):
            raise ThreeDCTSError("invalid coordinated CTS buffer inventory")
        if baseline_clock_buffers == 0:
            buffer_ratio = 1.0 if coordinated_clock_buffers == 0 else None
            resource_guard_passed = coordinated_clock_buffers == 0
        else:
            buffer_ratio = coordinated_clock_buffers / baseline_clock_buffers
            resource_guard_passed = (
                coordinated_clock_buffers <= 2 * baseline_clock_buffers
            )

    improvement = baseline_objective - coordinated_objective
    selected = (
        "coordinated"
        if improvement > tolerance_ns and resource_guard_passed
        else "baseline"
    )
    if selected == "coordinated":
        reason = "strict-same-domain-skew-improvement"
    elif not resource_guard_passed:
        reason = "clock-buffer-2x-guard"
    else:
        reason = "no-improvement-beyond-numerical-tolerance"
    return {
        "selected": selected,
        "reason": reason,
        "comparison_tolerance_ns": tolerance_ns,
        "baseline_max_per_clock_skew_ns": baseline_objective,
        "coordinated_max_per_clock_skew_ns": coordinated_objective,
        "improvement_ns": improvement,
        "clock_inventory": inventory,
        "exact_clock_sink_identity_match": baseline_inventory is not None,
        "resource_guard": {
            "available": baseline_clock_buffers is not None,
            "baseline_clock_buffers": baseline_clock_buffers,
            "coordinated_clock_buffers": coordinated_clock_buffers,
            "coordinated_to_baseline_ratio": buffer_ratio,
            "max_ratio": 2.0,
            "passed": resource_guard_passed,
            "area_basis": "same configured per-die clock-buffer masters",
        },
        "setup_hold_used_for_selection": False,
    }


_CLOCK_INTERFACE_COLUMNS = (
    "hbt_id",
    "point",
    "analysis",
    "transition",
    "available",
    "arrival_ns",
    "slew_ns",
    "downstream_cap_pf",
    "path_resistance_ui",
    "elmore_ns",
)


def _read_passive_clock_interface(
    path: Path, expected_hbt_id: str
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != _CLOCK_INTERFACE_COLUMNS:
        raise ThreeDCTSError("invalid passive clock-interface header")
    records: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(_CLOCK_INTERFACE_COLUMNS):
            raise ThreeDCTSError("invalid passive clock-interface row")
        raw = dict(zip(_CLOCK_INTERFACE_COLUMNS, fields))
        point = raw["point"]
        analysis = raw["analysis"]
        transition = raw["transition"]
        if (
            raw["hbt_id"] != expected_hbt_id
            or point not in {"near", "far"}
            or analysis not in {"max", "min"}
            or transition not in {"rise", "fall"}
            or raw["available"] != "1"
        ):
            raise ThreeDCTSError("incomplete passive clock-interface query")
        values = _finite_values(raw[name] for name in _CLOCK_INTERFACE_COLUMNS[5:])
        if len(values) != 5:
            raise ThreeDCTSError("non-finite passive clock-interface value")
        row = {
            name: float(raw[name]) for name in _CLOCK_INTERFACE_COLUMNS[5:]
        }
        transition_rows = records.setdefault(point, {}).setdefault(analysis, {})
        if transition in transition_rows:
            raise ThreeDCTSError("duplicate passive clock-interface row")
        transition_rows[transition] = row
    if set(records) != {"near", "far"} or any(
        set(analyses) != {"max", "min"}
        or any(set(transitions) != {"rise", "fall"} for transitions in analyses.values())
        for analyses in records.values()
    ):
        raise ThreeDCTSError("passive clock-interface query does not contain 8 rows")
    return records


def _passive_top_sink_edges(path: Path) -> Dict[str, List[float]]:
    edges = {
        "max_rise": [],
        "max_fall": [],
        "min_rise": [],
        "min_fall": [],
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 7:
            raise ThreeDCTSError("invalid passive clock-sink row")
        if fields[0] != "top":
            continue
        for key, raw in zip(edges, fields[3:]):
            values = _finite_values((raw,))
            if values:
                edges[key].append(values[0])
    if any(not values for values in edges.values()):
        raise ThreeDCTSError("passive view has incomplete top clock-sink arrivals")
    return edges


def _derive_passive_clock_proxy(
    passive: Path, clock: HBTBoundary, clock_period_ns: float
) -> Dict[str, object]:
    """Derive the source CTS load proxy from one continuous passive snapshot."""

    interface = _read_passive_clock_interface(
        passive / "clock_hbt_interface.tsv", clock.hbt_id
    )
    sinks = _passive_top_sink_edges(passive / "clock_sinks.tsv")
    if clock_period_ns <= 0.0:
        raise ThreeDCTSError("passive clock proxy requires a positive period")

    # Clock-pin properties can be reported one waveform cycle away from the
    # cutpoint.  Normalize each same-edge delta into the master-clock period
    # before reducing the remote-tree min/max range.
    def normalized(values: Iterable[float], origin: float) -> List[float]:
        result = [(value - origin) % clock_period_ns for value in values]
        if not result:
            raise ThreeDCTSError("passive clock proxy has no downstream sinks")
        return result

    max_rise = normalized(
        sinks["max_rise"], interface["near"]["max"]["rise"]["arrival_ns"]
    )
    max_fall = normalized(
        sinks["max_fall"], interface["near"]["max"]["fall"]["arrival_ns"]
    )
    min_rise = normalized(
        sinks["min_rise"], interface["near"]["min"]["rise"]["arrival_ns"]
    )
    min_fall = normalized(
        sinks["min_fall"], interface["near"]["min"]["fall"]["arrival_ns"]
    )
    delays = {
        "max_rise_ns": max(max_rise),
        "max_fall_ns": max(max_fall),
        "min_rise_ns": min(min_rise),
        "min_fall_ns": min(min_fall),
    }
    downstream_min = min(delays["min_rise_ns"], delays["min_fall_ns"])
    downstream_max = max(delays["max_rise_ns"], delays["max_fall_ns"])
    if downstream_max < downstream_min:
        raise ThreeDCTSError("passive clock proxy has an inverted delay range")
    capacitance_pf = max(
        interface["near"][analysis][transition]["downstream_cap_pf"]
        for analysis in ("max", "min")
        for transition in ("rise", "fall")
    )
    if capacitance_pf <= 0.0:
        raise ThreeDCTSError("passive clock proxy has non-positive capacitance")
    return {
        "model": "continuous-passive-cutpoint",
        "source_visible_clock_capacitance_pf": capacitance_pf,
        "downstream_min_ns": downstream_min,
        "downstream_max_ns": downstream_max,
        "edge_delays": delays,
        "near_cutpoint": interface["near"],
    }


def _derive_passive_clock_handoff(
    passive: Path, clock: HBTBoundary, clock_period_ns: float
) -> Dict[str, object]:
    """Derive receiving CTS source arrival/slew at the passive far cutpoint."""

    interface = _read_passive_clock_interface(
        passive / "clock_hbt_interface.tsv", clock.hbt_id
    )
    far = interface["far"]["max"]
    source_latency = max(
        far["rise"]["arrival_ns"],
        far["fall"]["arrival_ns"] - clock_period_ns / 2.0,
    )
    source_transition = max(far["rise"]["slew_ns"], far["fall"]["slew_ns"])
    if source_latency < -1.0e-6 or source_transition <= 0.0:
        raise ThreeDCTSError("invalid passive receiving clock handoff")
    return {
        "model": "continuous-passive-cutpoint",
        "source_latency_ns": max(0.0, source_latency),
        "source_transition_ns": source_transition,
        "far_cutpoint": interface["far"],
    }


_POST_CTS_CUTPOINT_COLUMNS = (
    "hbt_id",
    "point",
    "analysis",
    "transition",
    "available",
    "arrival_ns",
    "slew_ns",
    "required_ns",
    "slack_ns",
    "wire_delay_ns",
    "elmore_ns",
    "downstream_cap_ui",
    "path_resistance_ui",
    "downstream_load_count",
    "constrained_load_count",
    "constrained",
)


def _optional_finite(raw: str) -> Optional[float]:
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        return None
    return value


def _parse_post_cts_cutpoints(
    path: Path, handoff: OptHandoff
) -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    """Read complete max/min passive timing at every HBT cutpoint."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != _POST_CTS_CUTPOINT_COLUMNS:
        raise ThreeDCTSError("invalid post-CTS passive cutpoint header")
    expected = {row.hbt_id for row in handoff.boundaries}
    records: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(_POST_CTS_CUTPOINT_COLUMNS):
            raise ThreeDCTSError("invalid post-CTS passive cutpoint row")
        raw = dict(zip(_POST_CTS_CUTPOINT_COLUMNS, fields))
        hbt_id = raw["hbt_id"]
        point = raw["point"]
        analysis = raw["analysis"]
        transition = raw["transition"]
        if (
            hbt_id not in expected
            or point not in {"near", "far"}
            or analysis not in {"max", "min"}
            or transition not in {"rise", "fall"}
        ):
            raise ThreeDCTSError("invalid post-CTS passive cutpoint identity")
        row: Dict[str, object] = {
            key: _optional_finite(raw[key])
            for key in _POST_CTS_CUTPOINT_COLUMNS[5:13]
        }
        row.update({
            "available": raw["available"] == "1",
            "downstream_load_count": int(raw["downstream_load_count"]),
            "constrained_load_count": int(raw["constrained_load_count"]),
            "constrained": raw["constrained"] == "1",
        })
        transition_rows = (
            records.setdefault(hbt_id, {})
            .setdefault(point, {})
            .setdefault(analysis, {})
        )
        if transition in transition_rows:
            raise ThreeDCTSError("duplicate post-CTS passive cutpoint row")
        transition_rows[transition] = row
    if set(records) != expected or any(
        set(points) != {"near", "far"}
        or any(
            set(analyses) != {"max", "min"}
            or any(
                set(transitions) != {"rise", "fall"}
                for transitions in analyses.values()
            )
            for analyses in points.values()
        )
        for points in records.values()
    ):
        raise ThreeDCTSError(
            "post-CTS passive cutpoints do not cover max/min rise/fall for every HBT"
        )
    return records


def _post_cts_boundary_budget_rows(
    handoff: OptHandoff,
    cutpoints: Mapping[
        str, Mapping[str, Mapping[str, Mapping[str, object]]]
    ],
    clocks: Mapping[str, float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for boundary in handoff.boundaries:
        source: Dict[str, object] = {}
        sink: Dict[str, object] = {}
        for analysis in ("max", "min"):
            for transition in ("rise", "fall"):
                near = cutpoints[boundary.hbt_id]["near"][analysis][transition]
                far = cutpoints[boundary.hbt_id]["far"][analysis][transition]
                source["required_{}_{}".format(analysis, transition)] = (
                    near["required_ns"] if near["constrained"] else None
                )
                sink["arrival_{}_{}".format(analysis, transition)] = (
                    far["arrival_ns"] if far["available"] else None
                )
                sink["slew_{}_{}".format(analysis, transition)] = (
                    far["slew_ns"] if far["available"] else None
                )
        rows.append({
            "hbt_id": boundary.hbt_id,
            "net_name": boundary.net_name,
            "source_die": boundary.source_die,
            "sink_die": boundary.sink_die,
            "source_port": boundary.source_port,
            "sink_port": boundary.sink_port,
            "direction": boundary.direction,
            "is_clock": boundary.is_clock,
            "clocks": list(clocks) if boundary.is_clock else [],
            "source": source,
            "sink": sink,
        })
    return rows


def _post_cts_boundary_sdc(
    die: str,
    base_sdc: Path,
    snapshot: Mapping[str, object],
    handoff: OptHandoff,
    target: Path,
) -> Dict[str, int]:
    """Add passive data-boundary budgets without replacing the real CTS clock."""

    period = _clock_period_ns(base_sdc)
    transition_default = float(handoff.timing_model["default_transition_ns"])
    lines = [
        base_sdc.read_text(encoding="utf-8").rstrip(),
        "\n\n# Open3DBench passive post-CTS data-boundary constraints.\n",
        "create_clock -name OPEN3D_BOUNDARY_REF -period {:.9f}\n".format(period),
    ]
    counts = {
        "source_loads": 0,
        "source_loads_unavailable": 0,
        "source_required": 0,
        "source_required_max": 0,
        "source_required_min": 0,
        "sink_arrivals": 0,
        "sink_arrivals_max": 0,
        "sink_arrivals_min": 0,
    }
    for row in snapshot["boundaries"]:
        if row["source_die"] == die:
            load = snapshot["passive_loads"][row["hbt_id"]][
                "remote_load_capacitance_ui"
            ]
            if float(load) > 0.0:
                lines.append(
                    "set_load {:.9f} [get_ports {{{}}}]\n".format(
                        float(load), row["source_port"]
                    )
                )
                counts["source_loads"] += 1
            else:
                counts["source_loads_unavailable"] += 1
            if not row["is_clock"]:
                for edge in ("rise", "fall"):
                    for analysis in ("max", "min"):
                        required = row["source"][
                            "required_{}_{}".format(analysis, edge)
                        ]
                        if required is None:
                            continue
                        delay = (
                            period - float(required)
                            if analysis == "max"
                            else -float(required)
                        )
                        lines.append(
                            "set_output_delay -clock OPEN3D_BOUNDARY_REF -{} -{} {:.9f} "
                            "[get_ports {{{}}}]\n".format(
                                analysis, edge, delay, row["source_port"]
                            )
                        )
                        counts["source_required"] += 1
                        counts["source_required_" + analysis] += 1
        elif not row["is_clock"]:
            for edge in ("rise", "fall"):
                for analysis in ("max", "min"):
                    slew = row["sink"][
                        "slew_{}_{}".format(analysis, edge)
                    ]
                    if slew is None or float(slew) <= 0.0:
                        slew = transition_default
                    lines.append(
                        "set_input_transition -{} -{} {:.9f} "
                        "[get_ports {{{}}}]\n".format(
                            analysis, edge, float(slew), row["sink_port"]
                        )
                    )
                    arrival = row["sink"][
                        "arrival_{}_{}".format(analysis, edge)
                    ]
                    if arrival is None:
                        continue
                    lines.append(
                        "set_input_delay -clock OPEN3D_BOUNDARY_REF -{} -{} {:.9f} "
                        "[get_ports {{{}}}]\n".format(
                            analysis, edge, float(arrival), row["sink_port"]
                        )
                    )
                    counts["sink_arrivals"] += 1
                    counts["sink_arrivals_" + analysis] += 1
    target.write_text("".join(lines), encoding="utf-8")
    return counts


def _dump_placement(
    die_odbs: Mapping[str, Path], output: Path, openroad: str
) -> Path:
    parts = []
    for die in ("bottom", "top"):
        part = output / (die + "_placement.tsv")
        _run_openroad(
            PLACEMENT_DUMP_TCL,
            {
                "OPT_DIE": die,
                "OPT_INPUT_ODB": str(die_odbs[die]),
                "OPT_PLACEMENT_TSV": str(part),
            },
            output / ("placement_" + die + ".log"),
            openroad,
        )
        parts.append(part.read_text(encoding="utf-8"))
    target = output / "placement.tsv"
    target.write_text("".join(parts), encoding="utf-8")
    return target


def _global_sta(
    handoff: OptHandoff,
    assembly: Path,
    die_odbs: Mapping[str, Path],
    die_verilogs: Mapping[str, Path],
    platform: Path,
    output: Path,
    openroad: str,
    query_all_boundaries: bool = False,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    from .three_d_opt import _boundary_plan

    output.mkdir(parents=True, exist_ok=True)
    plan = output / "boundary_plan.tsv"
    _boundary_plan(handoff, plan)
    lefs, libs = _platform_files(platform)
    _validate_netlist_liberty(assembly / "design.v", [*libs, assembly / "hbt.lib"])
    placement = _dump_placement(die_odbs, output, openroad)
    construction_env = {
            "CTS_TECH_LEF": str(platform / "lef/NangateOpenCellLibrary.tech.lef"),
            "CTS_DIE_LEFS": "\n".join(str(path) for path in lefs),
            "CTS_LIBS": "\n".join(str(path) for path in libs),
            "CTS_HBT_LEF": str(assembly / "hbt_timing.lef"),
            "CTS_HBT_LIB": str(assembly / "hbt.lib"),
            "CTS_ASSEMBLY_V": str(assembly / "design.v"),
            "CTS_ASSEMBLY_SDC": str(assembly / "design.sdc"),
            "CTS_SET_RC": str(platform / "setRC.tcl"),
            "OPEN3D_LIBERTY_SUPPRESSIONS": str(
                platform / "liberty_suppressions.tcl"
            ),
            "CTS_ESTIMATED_SPEF": str(output / "placement_estimated.split.spef"),
            "CTS_PLACEMENT_TSV": str(placement),
            "CTS_BOUNDARY_PLAN": str(plan),
        }
    _run_openroad(
        CTS_RC_CONSTRUCTION_TCL,
        construction_env,
        output / "rc_construction.log",
        openroad,
    )
    estimated_spef = output / "placement_estimated.split.spef"
    if not estimated_spef.is_file() or estimated_spef.stat().st_size == 0:
        raise ThreeDCTSError("CTS RC construction did not write estimated SPEF")

    passive = output / "passive"
    passive.mkdir()
    links = passive_links(handoff.boundaries)
    write_passive_assembly(
        passive / "design.v",
        die_verilogs["bottom"],
        die_verilogs["top"],
        links,
    )
    shutil.copy2(assembly / "design.sdc", passive / "design.sdc")
    plan = stitch_estimated_spef(
        estimated_spef,
        passive / "design.spef",
        links,
        resistance_ohm=float(handoff.timing_model["resistance_ohm"]),
        capacitance_pf=float(handoff.timing_model["capacitance_pf"]),
    )
    write_passive_plan(passive / "plan.tsv", plan)
    _validate_netlist_liberty(passive / "design.v", libs)
    passive_env = {
        "CTS_TECH_LEF": construction_env["CTS_TECH_LEF"],
        "CTS_DIE_LEFS": construction_env["CTS_DIE_LEFS"],
        "CTS_LIBS": construction_env["CTS_LIBS"],
        "OPEN3D_LIBERTY_SUPPRESSIONS": construction_env[
            "OPEN3D_LIBERTY_SUPPRESSIONS"
        ],
        "CTS_PASSIVE_V": str(passive / "design.v"),
        "CTS_PASSIVE_SDC": str(passive / "design.sdc"),
        "CTS_PASSIVE_SPEF": str(passive / "design.spef"),
        "CTS_PASSIVE_PLAN": str(passive / "plan.tsv"),
        "CTS_CLOCK_HBT_ID": _clock_boundaries(handoff)[0].hbt_id,
        "CTS_PRIMARY_CLOCK_NAME": _physical_clock_name(
            _clean_sdc(handoff.dies["bottom"].sdc)
        ),
        "CTS_CLOCK_INTERFACE_TSV": str(passive / "clock_hbt_interface.tsv"),
        "CTS_QUERY_ALL_BOUNDARIES": "1" if query_all_boundaries else "0",
        "CTS_BOUNDARY_CUTPOINTS_TSV": str(passive / "boundary_cutpoints.tsv"),
        "CTS_BOUNDARY_LOADS_TSV": str(passive / "boundary_loads.tsv"),
        "CTS_EXPECTED_HBTS": str(len(handoff.boundaries)),
        "CTS_SINKS_TSV": str(passive / "clock_sinks.tsv"),
        "CTS_SINKS_BY_CLOCK_TSV": str(passive / "clock_sinks_by_clock.tsv"),
        "CTS_METRICS_TSV": str(passive / "metrics.tsv"),
        "CTS_CHECK_SETUP_REPORT": str(passive / "check_setup.rpt"),
        "CTS_SETUP_REPORT": str(passive / "setup.rpt"),
        "CTS_HOLD_REPORT": str(passive / "hold.rpt"),
        "CTS_ANNOTATION_REPORT": str(passive / "parasitic_annotation.rpt"),
    }
    _run_openroad(
        PASSIVE_CTS_STA_TCL,
        passive_env,
        passive / "sta.log",
        openroad,
    )
    passive_validation = _environment_script(passive_env, PASSIVE_CTS_STA_TCL)
    (passive / "validate_sta.tcl").write_text(passive_validation, encoding="utf-8")
    (assembly / "validate_sta.tcl").write_text(passive_validation, encoding="utf-8")
    passive_metrics = _read_key_values(passive / "metrics.tsv")
    passive_skew = _assembly_skew(passive / "clock_sinks.tsv")
    if query_all_boundaries:
        cutpoints = _parse_post_cts_cutpoints(
            passive / "boundary_cutpoints.tsv", handoff
        )
        loads = {}
        for line in (passive / "boundary_loads.tsv").read_text(
            encoding="utf-8"
        ).splitlines():
            hbt_id, raw_load = line.split("\t")
            loads[hbt_id] = {"remote_load_capacitance_ui": float(raw_load)}
        if set(loads) != {row.hbt_id for row in handoff.boundaries}:
            raise ThreeDCTSError("passive boundary loads do not cover every HBT")
        period = _clock_period_ns(assembly / "design.sdc")
        snapshot = {
            "schema_version": 2,
            "stage": "post-cts-passive-boundary-snapshot",
            "metrics": passive_metrics,
            "clocks": {"CLK": period},
            "boundaries": _post_cts_boundary_budget_rows(
                handoff, cutpoints, {"CLK": period}
            ),
            "passive_loads": loads,
        }
        _write_json(output / "boundary_snapshot.json", snapshot)
    _write_json(
        output / "timing.json",
        {
            "schema_version": 1,
            "authoritative_timing_view": "continuous-passive-hbt",
            "metrics": passive_metrics,
            "skew": passive_skew,
            "rc_construction": "directional-placeholder-only",
        },
    )
    return passive_metrics, passive_skew


def _assembly_cts_sta(
    handoff: OptHandoff,
    bottom_odb: Path,
    top_odb: Path,
    bottom_verilog: Path,
    top_verilog: Path,
    platform: Path,
    output: Path,
    openroad: str,
    query_all_boundaries: bool = False,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    _rebuild_assembly(handoff, bottom_verilog, top_verilog, output)
    (output / "design.sdc").write_text(
        _clean_sdc(handoff.assembly_sdc), encoding="utf-8"
    )
    return _global_sta(
        handoff,
        output,
        {"bottom": bottom_odb, "top": top_odb},
        {"bottom": bottom_verilog, "top": top_verilog},
        platform,
        output,
        openroad,
        query_all_boundaries,
    )


def run_cts(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    handoff = load_cts_handoff(Path(args.checkpoint))
    clocks = _clock_boundaries(handoff)
    clock = clocks[0]
    clock_period = _clock_period_ns(handoff.dies["bottom"].sdc)
    platform = Path(args.platform_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.json").exists():
        raise ThreeDCTSError("refusing to overwrite completed CTS checkpoint")
    if not 0.0 <= args.target_skew_ns or args.cluster_size == 1 or args.cluster_size < 0:
        raise ThreeDCTSError("invalid CTS target or clustering configuration")
    if args.cluster_diameter < 0.0:
        raise ThreeDCTSError("cluster diameter must be non-negative")
    if not 0.0 <= args.delay_buffer_derate <= 1.0:
        raise ThreeDCTSError("delay_buffer_derate must be in [0, 1]")
    work = output / "iterations"
    def run_die(*values: object, **keywords: object):
        die = str(values[1])
        keywords.setdefault(
            "signal_layer",
            args.bottom_signal_layer if die == "bottom" else args.top_signal_layer,
        )
        keywords.setdefault(
            "clock_layer",
            args.bottom_clock_layer if die == "bottom" else args.top_clock_layer,
        )
        return _run_die_cts(*values, **keywords)

    preview = work / "00_top_preview"
    preview_metrics, preview_latency, _unused = run_die(
        handoff,
        "top",
        handoff.dies["top"].odb,
        _receiving_sdc(
            handoff.dies["top"].sdc,
            clock,
            0.0,
            float(handoff.timing_model["default_transition_ns"]),
            clocks,
        ),
        platform,
        preview,
        args.openroad,
        args.top_buffer,
        args.cluster_size,
        args.cluster_diameter,
        args.delay_buffer_derate,
        clock,
        clock_period_ns=clock_period,
    )

    # Embed the top preview in the real passive assembly before constructing
    # the temporary bottom CTS load.  The near cutpoint sees exactly the HBT pi
    # section plus the previewed remote tree; no fixed delay or local-root-cap
    # approximation is allowed into the proxy parameters.
    proxy_assembly = work / "01_passive_proxy"
    _proxy_global, _proxy_skew = _assembly_cts_sta(
        handoff,
        handoff.dies["bottom"].odb,
        preview / "design.odb",
        handoff.dies["bottom"].verilog,
        preview / "design.v",
        platform,
        proxy_assembly,
        args.openroad,
    )
    seed_interface = _read_passive_clock_interface(
        proxy_assembly / "passive/clock_hbt_interface.tsv", clock.hbt_id
    )
    source_visible_clock_capacitance_pf = max(
        seed_interface["near"][analysis][transition]["downstream_cap_pf"]
        for analysis in ("max", "min")
        for transition in ("rise", "fall")
    )
    modeled_source_visible_clock_capacitance_pf = source_visible_clock_capacitance_pf

    models = output / "models"
    models.mkdir()
    seed_load_lib = models / "clock_hbt_seed.lib"
    seed_load_lef = models / "clock_hbt_seed.lef"
    load_lib = models / "clock_hbt.lib"
    load_lef = models / "clock_hbt.lef"
    load_cell = _write_clock_load_models(
        seed_load_lib,
        seed_load_lef,
        clock,
        source_visible_clock_capacitance_pf,
        0.0,
        0.0,
        args.bottom_hbt_layer,
    )

    bottom_seed = work / "02_bottom_seed"
    _seed_metrics, _seed_latency, _seed_handoff = run_die(
        handoff,
        "bottom",
        handoff.dies["bottom"].odb,
        _source_sdc(
            handoff.dies["bottom"].sdc,
            clocks,
            float(handoff.timing_model["default_transition_ns"]),
        ),
        platform,
        bottom_seed,
        args.openroad,
        args.bottom_buffer,
        args.cluster_size,
        args.cluster_diameter,
        args.delay_buffer_derate,
        clock,
        load_cell,
        seed_load_lib,
        seed_load_lef,
    )

    # The two finished local trees now give the passive graph a realistic
    # source slew and remote load.  This query is reporting/boundary evidence;
    # it does not trigger another CTS pass.
    tuned_proxy_assembly = work / "03_passive_proxy_tuned"
    _tuned_proxy_global, _tuned_proxy_skew = _assembly_cts_sta(
        handoff,
        bottom_seed / "design.odb",
        preview / "design.odb",
        bottom_seed / "design.v",
        preview / "design.v",
        platform,
        tuned_proxy_assembly,
        args.openroad,
        query_all_boundaries=args.repair_hold,
    )
    passive_proxy = _derive_passive_clock_proxy(
        tuned_proxy_assembly / "passive", clock, clock_period
    )
    downstream_min = float(passive_proxy["downstream_min_ns"])
    downstream_max = float(passive_proxy["downstream_max_ns"])
    source_visible_clock_capacitance_pf = float(
        passive_proxy["source_visible_clock_capacitance_pf"]
    )
    # Preserve the qualified one-pass result as a physical fallback.  Now put
    # the complete passive HBT + receiving-tree delay back on the temporary
    # source-side leaf.  TritonCTS treats that leaf as a macro clock endpoint
    # with insertion delay, repairs long clock wires, and uses its native
    # latency balancer to add real delay buffers only to the faster local
    # register side of the split.
    bottom_baseline = bottom_seed
    baseline_bottom_metrics = _seed_metrics
    baseline_bottom_latency = _seed_latency
    baseline_assembly = tuned_proxy_assembly
    baseline_global = _tuned_proxy_global
    baseline_skew = _tuned_proxy_skew

    _write_clock_load_models(
        load_lib,
        load_lef,
        clock,
        source_visible_clock_capacitance_pf,
        downstream_min,
        downstream_max,
        args.bottom_hbt_layer,
    )
    bottom_coordinated = work / "04_bottom_coordinated"
    (
        coordinated_bottom_metrics,
        coordinated_bottom_latency,
        _coordinated_handoff,
    ) = run_die(
        handoff,
        "bottom",
        handoff.dies["bottom"].odb,
        _source_sdc(
            handoff.dies["bottom"].sdc,
            clocks,
            float(handoff.timing_model["default_transition_ns"]),
        ),
        platform,
        bottom_coordinated,
        args.openroad,
        args.bottom_buffer,
        args.cluster_size,
        args.cluster_diameter,
        args.delay_buffer_derate,
        clock,
        load_cell,
        load_lib,
        load_lef,
    )
    coordinated_assembly = work / "05_passive_coordinated"
    coordinated_global, coordinated_skew = _assembly_cts_sta(
        handoff,
        bottom_coordinated / "design.odb",
        preview / "design.odb",
        bottom_coordinated / "design.v",
        preview / "design.v",
        platform,
        coordinated_assembly,
        args.openroad,
        query_all_boundaries=args.repair_hold,
    )
    alignment_selection = _select_coordinated_skew(
        baseline_skew,
        coordinated_skew,
        baseline_inventory=_clock_domain_inventory(
            baseline_assembly / "passive/clock_sinks_by_clock.tsv"
        ),
        coordinated_inventory=_clock_domain_inventory(
            coordinated_assembly / "passive/clock_sinks_by_clock.tsv"
        ),
        # Only the bottom tree changes.  Applying the 2x guard to that die is
        # stricter than the required whole-design count/area guard because the
        # configured buffer master is unchanged and the top tree is identical.
        baseline_clock_buffers=int(
            baseline_bottom_metrics["clock_buffer_count"]
        ),
        coordinated_clock_buffers=int(
            coordinated_bottom_metrics["clock_buffer_count"]
        ),
    )
    if alignment_selection["selected"] == "coordinated":
        selected_bottom = bottom_coordinated
        selected_assembly = coordinated_assembly
        selected_global = coordinated_global
        selected_skew = coordinated_skew
        bottom_metrics = coordinated_bottom_metrics
        bottom_latency = coordinated_bottom_latency
    else:
        selected_bottom = bottom_baseline
        selected_assembly = baseline_assembly
        selected_global = baseline_global
        selected_skew = baseline_skew
        bottom_metrics = baseline_bottom_metrics
        bottom_latency = baseline_bottom_latency

    # The receiving topology is unchanged by source-side alignment.  Re-query
    # its far cutpoint from the selected physical pair and publish that arrival
    # and slew in the die-local SDC.  This is an artifact handoff only; the
    # continuous passive assembly remains the timing authority.
    passive_handoff = _derive_passive_clock_handoff(
        selected_assembly / "passive", clock, clock_period
    )
    clock_handoff = passive_handoff
    top_source_latency = float(passive_handoff["source_latency_ns"])
    top_source_transition = float(passive_handoff["source_transition_ns"])
    selected_top = preview
    top_metrics = preview_metrics
    top_latency = preview_latency
    write_sdc_projection(selected_top / "design.sdc", project_sdc_for_handoff(
        handoff, "top", _receiving_sdc(
            handoff.dies["top"].sdc,
            clock,
            top_source_latency,
            top_source_transition,
            clocks,
        ), source_label="CTS final top HBT latency update",
        def_paths={"bottom": selected_bottom / "design.def", "top": selected_top / "design.def"},
        odb_paths={"bottom": selected_bottom / "design.odb", "top": selected_top / "design.odb"},
        platform=platform, openroad=args.openroad,
        evidence_dir=selected_top / "final_sdc_evidence",
    ))

    # CTS owns the two clock trees, not data-path optimization.  The completed
    # two-tree passive assembly is already the final timing authority when hold
    # repair is disabled.  Only explicit hold repair pays to export every data
    # boundary cutpoint and to rebuild timing after physical mutation.
    pre_repair_assembly = selected_assembly
    _pre_repair_global = selected_global
    _pre_repair_skew = selected_skew
    _mid_repair_global = _pre_repair_global
    bottom_repair_metrics = {
        "enabled": False,
        "status": "skipped",
        "reason": "hold repair is opt-in",
    }
    top_repair_metrics = dict(bottom_repair_metrics)
    if args.repair_hold:
        pre_repair_snapshot = json.loads(
            (pre_repair_assembly / "boundary_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        bottom_repair = work / "06_bottom_hold_repair"
        bottom_repair_metrics = _post_cts_hold_repair_die(
            handoff,
            "bottom",
            selected_bottom / "design.odb",
            selected_bottom / "design.sdc",
            pre_repair_snapshot,
            platform,
            bottom_repair,
            args.openroad,
            args.bottom_signal_layer,
            args.bottom_clock_layer,
        )

        mid_repair_assembly = work / "07_assembly_after_bottom_hold"
        _mid_repair_global, _mid_repair_skew = _assembly_cts_sta(
            handoff,
            bottom_repair / "design.odb",
            selected_top / "design.odb",
            bottom_repair / "design.v",
            selected_top / "design.v",
            platform,
            mid_repair_assembly,
            args.openroad,
            query_all_boundaries=True,
        )
        mid_repair_snapshot = json.loads(
            (mid_repair_assembly / "boundary_snapshot.json").read_text(encoding="utf-8")
        )
        top_repair = work / "08_top_hold_repair"
        top_repair_metrics = _post_cts_hold_repair_die(
            handoff,
            "top",
            selected_top / "design.odb",
            selected_top / "design.sdc",
            mid_repair_snapshot,
            platform,
            top_repair,
            args.openroad,
            args.top_signal_layer,
            args.top_clock_layer,
        )
        selected_bottom = bottom_repair
        selected_top = top_repair

    bottom = output / "bottom"
    top = output / "top"
    shutil.copytree(selected_bottom, bottom, copy_function=copy_with_log_reference)
    shutil.copytree(selected_top, top, copy_function=copy_with_log_reference)
    # Rebuild the final audit against the selected physical databases, not
    # their pre-CTS or pre-resizer port/net inventory.
    for die, directory in (("bottom", bottom), ("top", top)):
        final_sdc = directory / "design.sdc"
        write_sdc_projection(final_sdc, project_sdc_for_handoff(
            handoff, die, final_sdc.read_text(encoding="utf-8"), source_label=str(final_sdc),
            def_paths={"bottom": bottom / "design.def", "top": top / "design.def"},
            odb_paths={"bottom": bottom / "design.odb", "top": top / "design.odb"},
            platform=platform, openroad=args.openroad,
            evidence_dir=output / "final_sdc_evidence",
        ))
    assembly = output / "assembly"
    if args.repair_hold:
        global_metrics, skew = _assembly_cts_sta(
            handoff,
            bottom / "design.odb",
            top / "design.odb",
            bottom / "design.v",
            top / "design.v",
            platform,
            assembly,
            args.openroad,
        )
    else:
        # No physical state changed after selection.  Publish the already
        # validated passive graph verbatim instead of regenerating the same
        # placement projection and large stitched SPEF.
        if alignment_selection["selected"] == "coordinated":
            shutil.copytree(coordinated_assembly, assembly)
        else:
            shutil.copytree(baseline_assembly, assembly)
        global_metrics = selected_global
        skew = selected_skew
    source_root = handoff.checkpoint.parent
    for name in (
        "hbt.csv",
        "name_map.csv",
        "partition_assignment.csv",
        "timing_model.json",
        "problem.json",
        "assignment_coordinates.csv",
        "hbt_manifest.json",
    ):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, output / name)
    shutil.copy2(handoff.checkpoint.parent / "assembly/boundary.json", assembly / "boundary.json")

    warning_audit = _warning_summary(output)
    if warning_audit["unexpected_count"]:
        raise ThreeDCTSError("CTS output contains unexpected OpenROAD warnings")
    target_report = _skew_target_report(skew, args.target_skew_ns)
    metrics = {
        "schema_version": 1,
        "algorithm": "openroad-cross-die-balanced",
        "status": "success",
        "clock_hbt_count": 1,
        "clock_hbt": clock.hbt_id,
        "clock_hbt_model_status": handoff.timing_model["status"],
        "configuration": {
            "target_skew_ns": args.target_skew_ns,
            "cluster_size": args.cluster_size,
            "cluster_diameter_um": args.cluster_diameter,
            "clustering_mode": (
                "automatic"
                if args.cluster_size == 0 and args.cluster_diameter == 0.0
                else "explicit"
            ),
            "delay_buffer_derate": args.delay_buffer_derate,
            "repair_hold": args.repair_hold,
            "bottom_buffer": args.bottom_buffer,
            "top_buffer": args.top_buffer,
        },
        "hierarchical_insertion_delay": {
            "enabled": True,
            "preview_top": preview_latency,
            "parameter_source": "continuous-passive-cutpoint",
            "measured_downstream_min_ns": downstream_min,
            "measured_downstream_max_ns": downstream_max,
            "modeled_source_visible_clock_capacitance_pf": (
                modeled_source_visible_clock_capacitance_pf
            ),
            "measured_source_visible_clock_capacitance_pf": (
                source_visible_clock_capacitance_pf
            ),
            "proxy_query": passive_proxy,
            "applied_to_physical_source_tree": (
                alignment_selection["selected"] == "coordinated"
            ),
            "receiving_handoff_query": passive_handoff,
        },
        "bottom": {"cts": bottom_metrics, "latency": bottom_latency},
        "top": {"cts": top_metrics, "latency": top_latency},
        "cross_die_alignment": {
            "method": "passive-insertion-delay-native-latency-balancing",
            "construction_order": [
                "top-preview",
                "source-capacitance-seed",
                "passive-delay-measurement",
                "source-coordinated-rebuild",
                "authoritative-passive-selection",
            ],
            "tree_retries": 1,
            "clock_net_repair_before_latency_balance": True,
            "load_model": {
                "cell": load_cell,
                "seed_liberty": "models/clock_hbt_seed.lib",
                "coordinated_liberty": "models/clock_hbt.lib",
                "capacitance_pf": source_visible_clock_capacitance_pf,
                "downstream_min_ns": downstream_min,
                "downstream_max_ns": downstream_max,
            },
            "baseline": {
                "bottom_cts": baseline_bottom_metrics,
                "bottom_latency": baseline_bottom_latency,
                "timing": baseline_global,
                "skew": baseline_skew,
            },
            "coordinated": {
                "bottom_cts": coordinated_bottom_metrics,
                "bottom_latency": coordinated_bottom_latency,
                "timing": coordinated_global,
                "skew": coordinated_skew,
            },
            "selection": alignment_selection,
            "selected_bottom_clock_buffer_count": bottom_metrics.get(
                "clock_buffer_count"
            ),
            "root_alignment_buffer_count": max(
                0,
                int(bottom_metrics.get("clock_buffer_count", 0))
                - int(baseline_bottom_metrics.get("clock_buffer_count", 0)),
            ),
        },
        "post_cts_hold_repair": {
            "enabled": args.repair_hold,
            "sequence": ["bottom", "top"],
            "max_iterations_per_die": 1,
            "clock_tree_protected": True,
            "boundary_budget_model": "continuous-passive-cutpoint",
            "bottom": bottom_repair_metrics,
            "top": top_repair_metrics,
            "pre_repair_global_timing": _pre_repair_global,
            "after_bottom_global_timing": _mid_repair_global,
        },
        "clock_handoff": clock_handoff,
        "assembly": {"timing": global_metrics, "skew": skew},
        "target_skew_ns": args.target_skew_ns,
        "target_met": target_report["target_met"],
        "skew_target_evaluation": target_report,
        "parasitics": {
            "assembly": "continuous-passive-hbt-placement-estimated-rc",
            "per_die": "placement-estimated-rc",
            "hbt": "flat-pi-section-3ohm-0.6fF",
            "rc_construction": "directional-placeholder-only",
            "local_clock_load_proxy": "tritoncts-only-not-a-global-timing-view",
            "spef": "temporary-placement-estimated-flat-stitching-artifact",
        },
        "timing_validity": {
            "graph_completeness": "passed",
            "unclocked_or_unconstrained_endpoints": 0,
            "generated_clock_hbt_count": 0,
            "formal_clock_hbt_count": 1,
            "remote_clock_tree_load_visible_to_source_cts": True,
            "local_clock_proxy_parameters_from_passive_cutpoint": True,
            "receiving_clock_from_passive_far_cutpoint": True,
            "whole_path_latency_used_for_physical_balancing": True,
            "fixed_hbt_delay_in_cts_coordination": False,
            "hbt_is_passive_rc_in_authoritative_global_sta": True,
            "authoritative_timing_view": "continuous-passive-hbt",
            "global_adapter_timing_view": False,
            "assembly_clock_net_type_restored": True,
            "global_timing_role": "post-cts-placement-estimated",
            "signoff_ready": False,
        },
        "warning_audit": warning_audit,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "metrics.json", metrics)
    checkpoint = json.loads(handoff.checkpoint.read_text(encoding="utf-8"))
    source_checkpoint = os.path.relpath(handoff.checkpoint, output).replace(
        os.sep, "/"
    )
    source_opt_metrics = os.path.relpath(
        handoff.checkpoint.parent / "metrics.json", output
    ).replace(os.sep, "/")
    if isinstance(checkpoint.get("opt"), dict):
        checkpoint["opt"]["metrics"] = source_opt_metrics
    checkpoint.update(
        {
            "stage": "cts",
            "source_checkpoint": source_checkpoint,
            "dies": {
                die: {
                    "odb": die + "/design.odb",
                    "def": die + "/design.def",
                    "verilog": die + "/design.v",
                    "sdc": die + "/design.sdc",
                }
                for die in ("bottom", "top")
            },
            "assembly": {
                "verilog": "assembly/design.v",
                "sdc": "assembly/design.sdc",
                "liberty": "assembly/hbt.lib",
                "timing_lef": "assembly/hbt_timing.lef",
                "boundary": "assembly/boundary.json",
                "validation_tcl": "assembly/validate_sta.tcl",
                "passive": {
                    "verilog": "assembly/passive/design.v",
                    "spef": "assembly/passive/design.spef",
                    "plan": "assembly/passive/plan.tsv",
                    "validation_tcl": "assembly/passive/validate_sta.tcl",
                },
            },
            "cts": {
                "algorithm": "openroad-cross-die-balanced",
                "metrics": "metrics.json",
                "global_skew_ns": skew["global_skew_ns"],
                "global_skew_valid_as_single_clock": skew["global_skew_valid_as_single_clock"],
                "max_per_clock_skew_ns": skew.get("max_per_clock_skew_ns"),
                "clock_sinks_by_clock": "assembly/passive/clock_sinks_by_clock.tsv",
                "target_skew_ns": args.target_skew_ns,
                "target_met": metrics["target_met"],
                "target_scope": "each-independent-clock-domain",
            },
            "ready_for_route": True,
        }
    )
    _write_json(output / "checkpoint.json", checkpoint)
    validate_checkpoint(output / "checkpoint.json", require_files=True)
    return output / "checkpoint.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--platform-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--openroad", default="/usr/local/bin/openroad")
    parser.add_argument("--bottom-buffer", default="BUF_X4_bottom")
    parser.add_argument("--top-buffer", default="BUF_X4_top")
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=0,
        help="TritonCTS sink cluster size; zero selects automatic clustering",
    )
    parser.add_argument(
        "--cluster-diameter",
        type=float,
        default=0.0,
        help="maximum cluster diameter in um; zero selects automatic clustering",
    )
    parser.add_argument("--delay-buffer-derate", type=float, default=1.0)
    parser.add_argument("--target-skew-ns", type=float, default=0.10)
    parser.add_argument(
        "--repair-hold",
        action="store_true",
        help="run at most one bounded hold-repair pass per die; disabled by default",
    )
    parser.add_argument("--bottom-signal-layer", default="metal3")
    parser.add_argument("--top-signal-layer", default="metal3")
    parser.add_argument("--bottom-clock-layer", default="metal5")
    parser.add_argument("--top-clock-layer", default="metal5")
    parser.add_argument("--bottom-hbt-layer", default="metal10")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(run_cts(args))
    except (ContractError, ThreeDCTSError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit("3D CTS failed: {}".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
