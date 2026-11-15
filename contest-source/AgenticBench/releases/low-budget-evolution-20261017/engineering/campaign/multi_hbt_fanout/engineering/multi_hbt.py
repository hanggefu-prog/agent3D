"""Deterministic buffer-isolated multi-HBT synthesis and OpenDB materialization.

Context units are DBU, pF, ns, and ohm.  The model below only selects physical
actions; protected OpenDB readback and joint STA remain authoritative.
"""
from dataclasses import dataclass
import json, math, os, statistics, subprocess
from pathlib import Path

DIES = ("bottom", "top")
SEED = 1001


@dataclass(frozen=True)
class Buffer:
    name: str
    die: str
    input_pin: str
    output_pin: str
    input_cap: float
    max_cap: float
    max_fanout: int
    intrinsic: float
    resistance: float


@dataclass(frozen=True)
class Layout:
    sites: tuple
    receivers: tuple
    buffers: tuple
    worst: float


def number(value, label, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError("invalid " + label)
    return value


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer")
    return value


def name(value, label):
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\0"):
        raise ValueError("invalid " + label)
    return value


def names(values, label, nonempty=False):
    if not isinstance(values, list):
        raise ValueError(label + " must be a list")
    result = [name(v, label) for v in values]
    if len(result) != len(set(result)) or (nonempty and not result):
        raise ValueError("repeated/empty " + label)
    return result


def point(row, label):
    if not isinstance(row, dict):
        raise ValueError("missing " + label)
    return integer(row.get("x_dbu"), label + " x"), integer(row.get("y_dbu"), label + " y")


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def read_buffers(context):
    result = {d: [] for d in DIES}
    seen = set()
    if not isinstance(context.get("buffer_cells"), list):
        raise ValueError("buffer_cells must be a list")
    for row in context["buffer_cells"]:
        if not isinstance(row, dict) or row.get("die") not in DIES:
            raise ValueError("invalid buffer cell")
        die, master = row["die"], name(row.get("name"), "buffer master")
        if (die, master) in seen:
            raise ValueError("repeated buffer master")
        seen.add((die, master))
        if row.get("noninverting") is not True:
            continue
        result[die].append(Buffer(
            master, die, name(row.get("input_pin"), "buffer input"),
            name(row.get("output_pin"), "buffer output"),
            number(row.get("input_capacitance_pf"), "buffer input capacitance"),
            number(row.get("max_capacitance_pf"), "buffer max capacitance"),
            integer(row.get("max_fanout"), "buffer max fanout", 1),
            number(row.get("intrinsic_delay_ns"), "buffer intrinsic"),
            number(row.get("drive_resistance_ohm"), "buffer resistance")))
    for die in DIES:
        if not result[die]:
            raise ValueError("no noninverting " + die + " buffer")
        result[die] = tuple(sorted(result[die],
            key=lambda b: (b.intrinsic, b.resistance, b.input_cap, b.name)))
    return result


def validate(context):
    if not isinstance(context, dict) or context.get("schema") != "multi-hbt-context-v1":
        raise ValueError("unsupported multi-HBT context")
    if context.get("seed") != SEED:
        raise ValueError("multi-HBT synthesis requires seed 1001")
    if context.get("baseline_delay_pending_native_measurement") is not False:
        raise ValueError("unfinished baseline delays")
    expected_units = {"coordinate":"dbu","capacitance":"pf","time":"ns","resistance":"ohm"}
    if context.get("units") != expected_units:
        raise ValueError("unsupported units")
    integer(context.get("dbu_per_micron"), "DBU scale", 1)
    integer(context.get("new_hbt_id_base"), "HBT ID base")
    if not isinstance(context.get("limits"), dict):
        raise ValueError("missing limits")
    integer(context["limits"].get("max_added_hbts"), "HBT budget")
    native = context.get("native")
    if not isinstance(native, dict):
        raise ValueError("missing native metadata")
    for die in DIES:
        name(native.get(die+"_odb"), die+" ODB")
        name(native.get(die+"_hbt_master"), die+" HBT master")
        name(native.get(die+"_hbt_layer"), die+" HBT layer")
    for key in ("openroad_executable", "power_net", "ground_net"):
        name(native.get(key), key)
    integer(native.get("hbt_width_dbu"), "HBT width", 1)
    integer(native.get("hbt_height_dbu"), "HBT height", 1)
    disp = native.get("max_buffer_displacement_sites")
    if not isinstance(disp, list) or len(disp) != 2 or any(type(v) is not int or v < 0 for v in disp):
        raise ValueError("invalid site/row displacement")
    sites = context.get("site_geometry")
    if not isinstance(sites, dict):
        raise ValueError("missing site geometry")
    spare = names(context.get("spare_site_ids"), "spare site")
    for site in spare:
        point(sites.get(site), site)
        positions = sites[site].get("buffer_by_die")
        if not isinstance(positions, dict):
            raise ValueError("spare site lacks buffer positions")
        for die in DIES:
            point(positions.get(die), site+" "+die+" buffer")
    maps = [context.get(k) for k in ("target_nets","baseline_timing",
            "driver_geometry","receiver_geometry")]
    maps += [native.get(k) for k in ("driver_pins","receiver_pins","hbt_ports")]
    if not all(isinstance(x, dict) for x in maps) or not context["target_nets"]:
        raise ValueError("incomplete target/native mappings")
    targets, baseline, drivers, receivers, nd, nr, ports = maps
    for logical, target in targets.items():
        name(logical, "logical net")
        if (target.get("signal_type") != "data" or target.get("driver_count") != 1
                or target.get("is_clock") is not False or target.get("supported") is not True
                or target.get("timing_exceptions") != []):
            raise ValueError("ineligible target: " + logical)
        sd, td = target.get("source_die"), target.get("sink_die")
        if sd not in DIES or td not in DIES or sd == td:
            raise ValueError("invalid die ownership: " + logical)
        driver = name(target.get("original_driver"), "driver")
        if driver not in drivers or driver not in nd or drivers[driver].get("die") != sd or nd[driver].get("die") != sd:
            raise ValueError("invalid driver mapping: " + logical)
        point(drivers[driver], driver); name(nd[driver].get("iterm"), "driver ITerm")
        sink_ids = names(target.get("original_remote_receivers"), "receiver", True)
        for receiver in sink_ids:
            if receiver not in receivers or receiver not in nr or receivers[receiver].get("die") != td or nr[receiver].get("die") != td:
                raise ValueError("invalid receiver mapping: " + receiver)
            point(receivers[receiver], receiver)
            number(receivers[receiver].get("capacitance_pf"), "receiver capacitance")
            name(nr[receiver].get("iterm"), "receiver ITerm")
        hbt, site = target.get("original_hbt_id"), target.get("original_site_id")
        if hbt not in ports or site not in sites:
            raise ValueError("missing original HBT metadata: " + logical)
        name(ports[hbt].get("source_port"), "source port")
        name(ports[hbt].get("sink_port"), "sink port"); point(sites[site], site)
        delays = baseline.get(logical, {}).get("receiver_delays_ns")
        if not isinstance(delays, dict) or set(delays) != set(sink_ids):
            raise ValueError("baseline inventory mismatch: " + logical)
        for receiver, delay in delays.items():
            number(delay, "baseline delay "+receiver)
    model = context.get("timing_model")
    if not isinstance(model, dict):
        raise ValueError("missing timing model")
    for key in ("wire_delay_ns_per_dbu","hbt_delay_ns","driver_resistance_ohm",
                "min_predicted_improvement_ns"):
        number(model.get(key), key)
    number(context.get("reference_input_slew_ns"), "input slew")
    return read_buffers(context)


def force_nonempty(assign, scores, count):
    members = [sum(v == b for v in assign.values()) for b in range(count)]
    # Prevent a formally nonempty but physically useless one-load branch.
    minimum = max(1, len(assign) // (4 * count))
    for missing in range(count):
        while members[missing] < minimum:
            donors = {branch for branch in range(count) if members[branch] > minimum}
            if not donors:
                raise ValueError("not enough receivers for useful branches")
            choices = [(scores[r][missing]-scores[r][old], r, old)
                       for r, old in assign.items() if old in donors]
            _, receiver, old = min(choices)
            assign[receiver] = missing
            members[old] -= 1
            members[missing] += 1


def select_buffer(choices, ids, geometry, driver_r):
    ids = list(ids); cap = sum(float(geometry[r]["capacitance_pf"]) for r in ids)
    legal = [b for b in choices if len(ids) <= b.max_fanout and cap <= b.max_cap+1e-15]
    if not legal:
        return None
    return min(legal, key=lambda b: (b.intrinsic+.001*b.resistance*cap+
               .001*driver_r*b.input_cap, b.input_cap, b.name))


def evaluate(context, buffers, logical, added_sites):
    target = context["target_nets"][logical]
    ids = tuple(sorted(target["original_remote_receivers"]))
    if len(ids) < 1+len(added_sites):
        return None
    sd, geo, sites = target["source_die"], context["receiver_geometry"], context["site_geometry"]
    base = context["baseline_timing"][logical]["receiver_delays_ns"]
    model = context["timing_model"]; wire = float(model["wire_delay_ns_per_dbu"])
    hbt = float(model["hbt_delay_ns"]); driver_r = float(model["driver_resistance_ohm"])
    driver = point(context["driver_geometry"][target["original_driver"]], "driver")
    original = point(sites[target["original_site_id"]], "original")
    anchors = [point(sites[s], s) for s in added_sites]
    positions = [point(sites[s]["buffer_by_die"][sd], s+" buffer") for s in added_sites]
    common = statistics.median(float(base[r])-hbt-wire*(distance(driver,original)+
             distance(original,point(geo[r],r))) for r in ids)
    unloaded = min(b.intrinsic+.001*driver_r*b.input_cap for b in buffers[sd])
    scores = {}
    for receiver in ids:
        sink = point(geo[receiver], receiver)
        scores[receiver] = [float(base[receiver])] + [common+hbt+unloaded+wire*(
            distance(driver,p)+distance(p,a)+distance(a,sink))
            for a,p in zip(anchors,positions)]
    assign = {r:min(range(len(v)), key=lambda b:(v[b],b)) for r,v in scores.items()}
    force_nonempty(assign, scores, 1+len(added_sites))
    selected = []
    for _ in range(10):
        groups = [[r for r in ids if assign[r] == b] for b in range(1+len(added_sites))]
        selected = [select_buffer(buffers[sd], g, geo, driver_r) for g in groups[1:]]
        if any(b is None for b in selected):
            return None
        total = sum(float(geo[r]["capacitance_pf"]) for r in ids)
        original_cap = sum(float(geo[r]["capacitance_pf"]) for r in groups[0])
        delta = .001*driver_r*(original_cap+sum(b.input_cap for b in selected)-total)
        caps = [sum(float(geo[r]["capacitance_pf"]) for r in g) for g in groups]
        scores = {}
        for receiver in ids:
            sink = point(geo[receiver], receiver)
            row = [max(0.,float(base[receiver])+delta)]
            for i,(a,p,b) in enumerate(zip(anchors,positions,selected),1):
                row.append(max(0.,common+delta+hbt+b.intrinsic+.001*b.resistance*caps[i]+
                    wire*(distance(driver,p)+distance(p,a)+distance(a,sink))))
            scores[receiver] = row
        updated = {r:min(range(len(v)),key=lambda b:(v[b],b)) for r,v in scores.items()}
        force_nonempty(updated,scores,1+len(added_sites))
        if updated == assign: break
        assign = updated
    groups = tuple(tuple(sorted(r for r in ids if assign[r] == b))
                   for b in range(1+len(added_sites)))
    selected = tuple(select_buffer(buffers[sd],g,geo,driver_r) for g in groups[1:])
    total = sum(float(geo[r]["capacitance_pf"]) for r in ids)
    delta = .001*driver_r*(sum(float(geo[r]["capacitance_pf"]) for r in groups[0])+
                           sum(b.input_cap for b in selected)-total)
    worst = 0.
    for branch, group in enumerate(groups):
        cap = sum(float(geo[r]["capacitance_pf"]) for r in group)
        for receiver in group:
            if branch == 0: delay = float(base[receiver])+delta
            else:
                a,p,b = anchors[branch-1],positions[branch-1],selected[branch-1]
                delay = common+delta+hbt+b.intrinsic+.001*b.resistance*cap+wire*(
                    distance(driver,p)+distance(p,a)+distance(a,point(geo[receiver],receiver)))
            worst = max(worst,max(0.,delay))
    return Layout(tuple(added_sites),groups,selected,worst)


def choose(context, buffers):
    state = {}
    for logical,target in sorted(context["target_nets"].items()):
        ids = tuple(sorted(target["original_remote_receivers"]))
        worst = max(map(float,context["baseline_timing"][logical]["receiver_delays_ns"].values()))
        state[logical] = Layout((),(ids,),(),worst)
    budget = min(context["limits"]["max_added_hbts"],2*len(state),len(context["spare_site_ids"]))
    used = set(); floor = max(float(context["timing_model"]["min_predicted_improvement_ns"]),.002)
    for _ in range(budget):
        best = None
        for logical in sorted(state):
            current = state[logical]
            if len(current.sites) >= 2: continue
            for site in sorted(context["spare_site_ids"]):
                if site in used: continue
                candidate = evaluate(context,buffers,logical,current.sites+(site,))
                if candidate is None: continue
                key = (current.worst-candidate.worst,-len(candidate.sites),logical,site)
                if best is None or key > best[0]: best = (key,logical,site,candidate)
        if best is None or best[0][0] <= floor: break
        _,logical,site,candidate = best
        state[logical] = candidate; used.add(site)
    return state


def quote(value):
    value = str(value)
    if any(c in value for c in "\r\n\0"): raise ValueError("unsafe Tcl value")
    return '"'+"".join("\\"+c if c in '\\\\"$[]' else c for c in value)+'"'


def call(command,*values):
    return command+" "+" ".join(quote(v) for v in values)


PREFIX = r'''
set_thread_count 8
read_db $::env(MH_INPUT)
set db [ord::get_db]
set block [ord::get_db_block]
set tech [ord::get_db_tech]
set originals [dict create]
foreach inst [$block getInsts] {
  dict set originals [$inst getName] [list [$inst getLocation] [$inst getOrient] [$inst getPlacementStatus] [[$inst getMaster] getName]]
  $inst setPlacementStatus FIRM
}
set added_buffers 0
set added_anchors 0
proc require_iterm {name direction} {
  global block
  set pin [$block findITerm $name]
  if {$pin eq "NULL" || [[$pin getMTerm] getIoType] ne $direction} {error "invalid ITerm $name"}
  return $pin
}
proc create_net {name} {
  global block
  if {[$block findNet $name] ne "NULL"} {error "net collision $name"}
  set net [odb::dbNet_create $block $name]
  if {$net eq "NULL"} {error "net creation failed"}
  $net setSigType SIGNAL
}
proc create_anchor {hbt port net_name io master_name layer_name x y width height} {
  global db block tech added_anchors
  if {[$block findInst $hbt] ne "NULL" || [$block findBTerm $port] ne "NULL"} {error "HBT collision"}
  set net [$block findNet $net_name]
  set master [$db findMaster $master_name]
  set layer [$tech findLayer $layer_name]
  if {$net eq "NULL" || $master eq "NULL" || [$master getType] ne "COVER" || $layer eq "NULL"} {error "invalid HBT object"}
  set landing [$layer getWidth]
  set anchor [odb::dbInst_create $block $master $hbt]
  if {$anchor eq "NULL" || [llength [$anchor getITerms]] != 0} {error "invalid COVER"}
  $anchor setLocation $x $y
  $anchor setPlacementStatus FIRM
  set term [odb::dbBTerm_create $net $port]
  $term setIoType $io
  $term setSigType SIGNAL
  set bpin [odb::dbBPin_create $term]
  $bpin setPlacementStatus FIRM
  set lx [expr {$x+($width-$landing)/2}]
  set ly [expr {$y+($height-$landing)/2}]
  odb::dbBox_create $bpin $layer $lx $ly [expr {$lx+$landing}] [expr {$ly+$landing}]
  incr added_anchors
}
proc create_buffer {inst_name master_name input_name output_name driver_name net_name x y power ground} {
  global db block added_buffers
  if {[$block findInst $inst_name] ne "NULL"} {error "buffer collision"}
  set driver [require_iterm $driver_name OUTPUT]
  set trunk [$driver getNet]
  set net [$block findNet $net_name]
  set master [$db findMaster $master_name]
  if {$trunk eq "NULL" || $net eq "NULL" || $master eq "NULL"} {error "invalid buffer object"}
  set inst [odb::dbInst_create $block $master $inst_name]
  $inst setLocation $x $y
  $inst setPlacementStatus PLACED
  set input [$inst findITerm $input_name]
  set output [$inst findITerm $output_name]
  if {$input eq "NULL" || $output eq "NULL" || [[$input getMTerm] getIoType] ne "INPUT" || [[$output getMTerm] getIoType] ne "OUTPUT"} {error "invalid buffer pins"}
  $input connect $trunk
  $output connect $net
  set pg 0
  foreach pin [$inst getITerms] {
    set type [[$pin getMTerm] getSigType]
    if {$type ni {POWER GROUND}} {continue}
    set rail_name [expr {$type eq "POWER" ? $power : $ground}]
    set rail [$block findNet $rail_name]
    if {$rail eq "NULL"} {error "missing PG"}
    $pin connect $rail
    incr pg
  }
  if {$pg == 0} {error "buffer has no PG pins"}
  incr added_buffers
}
proc move_receiver {pin_name old_name new_name} {
  global block
  set pin [require_iterm $pin_name INPUT]
  set old [$pin getNet]
  set new [$block findNet $new_name]
  if {$old eq "NULL" || [$old getName] ne $old_name || $new eq "NULL"} {error "receiver net mismatch"}
  $pin disconnect
  $pin connect $new
}
'''


SUFFIX = r'''
if {$added_buffers != $::env(MH_BUFFERS) || $added_anchors != $::env(MH_ANCHORS)} {error "resource count mismatch"}
if {$added_buffers > 0} {detailed_placement -max_displacement [list $::env(MH_DX) $::env(MH_DY)]}
dict for {name old} $originals {
  set inst [$block findInst $name]
  if {$inst eq "NULL"} {error "original disappeared"}
  lassign $old xy orient status master
  if {[$inst getLocation] ne $xy || [$inst getOrient] ne $orient || [[$inst getMaster] getName] ne $master} {error "original changed"}
  $inst setPlacementStatus $status
  if {[$inst getPlacementStatus] ne $status} {error "status restore failed"}
}
set pg 0
foreach inst [$block getInsts] {
  foreach pin [$inst getITerms] {
    set type [[$pin getMTerm] getSigType]
    if {$type ni {POWER GROUND}} {continue}
    set expected [expr {$type eq "POWER" ? $::env(MH_POWER) : $::env(MH_GROUND)}]
    set actual [$pin getNet]
    if {$actual eq "NULL" || [$actual getName] ne $expected} {error "invalid PG"}
    incr pg
  }
}
check_placement -verbose
write_db $::env(MH_OUTPUT)
puts "MULTI_HBT_MATERIALIZED die=$::env(MH_DIE) buffers=$added_buffers anchors=$added_anchors pg=$pg"
'''


def build_script(context,die,physical):
    lines = [PREFIX]; buffers = anchors = 0; native = context["native"]
    for logical in sorted(physical):
        target = context["target_nets"][logical]
        if die == target["sink_die"]:
            port = native["hbt_ports"][target["original_hbt_id"]]["sink_port"]
            lines += ["set original_term [$block findBTerm %s]"%quote(port),
                'if {$original_term eq "NULL" || [$original_term getNet] eq "NULL"} {error "missing original sink port"}',
                "set original_sink_net [[$original_term getNet] getName]"]
            for receiver in target["original_remote_receivers"]:
                pin = native["receiver_pins"][receiver]["iterm"]
                lines += ["set check_pin [require_iterm %s INPUT]"%quote(pin),
                    'if {[$check_pin getNet] eq "NULL" || [[$check_pin getNet] getName] ne $original_sink_net} {error "receiver not on original net"}']
        for branch in physical[logical][1:]:
            net = branch["branch_id"]; site = context["site_geometry"][branch["site_id"]]
            lines.append(call("create_net",net))
            if die == target["source_die"]:
                cell = branch["buffer"]; pos = site["buffer_by_die"][die]
                lines.append(call("create_buffer",branch["buffer_instance"],cell.name,
                    cell.input_pin,cell.output_pin,
                    native["driver_pins"][target["original_driver"]]["iterm"],net,
                    pos["x_dbu"],pos["y_dbu"],native["power_net"],native["ground_net"]))
                buffers += 1; port,io = branch["hbt_id"]+"_SRC","OUTPUT"
            else:
                for receiver in branch["receivers"]:
                    lines.append("move_receiver %s $original_sink_net %s"%(
                        quote(native["receiver_pins"][receiver]["iterm"]),quote(net)))
                port,io = branch["hbt_id"]+"_DST","INPUT"
            lines.append(call("create_anchor",branch["hbt_id"],port,net,io,
                native[die+"_hbt_master"],native[die+"_hbt_layer"],site["x_dbu"],
                site["y_dbu"],native["hbt_width_dbu"],native["hbt_height_dbu"]))
            anchors += 1
        if die == target["sink_die"]:
            for receiver in physical[logical][0]["receivers"]:
                pin = native["receiver_pins"][receiver]["iterm"]
                lines += ["set kept [require_iterm %s INPUT]"%quote(pin),
                    'if {[$kept getNet] eq "NULL" || [[$kept getNet] getName] ne $original_sink_net} {error "kept receiver moved"}']
    return "\n".join(lines+[SUFFIX]),buffers,anchors


def run_native(context,output,die,physical):
    folder = output/die; folder.mkdir()
    script,buffers,anchors = build_script(context,die,physical)
    script_path = folder/"materialize.tcl"; script_path.write_text(script)
    odb = folder/"design.odb"; native = context["native"]; dx,dy = native["max_buffer_displacement_sites"]
    env = dict(os.environ); env.update(MH_INPUT=native[die+"_odb"],MH_OUTPUT=str(odb),
        MH_DIE=die,MH_BUFFERS=str(buffers),MH_ANCHORS=str(anchors),MH_DX=str(dx),
        MH_DY=str(dy),MH_POWER=native["power_net"],MH_GROUND=native["ground_net"])
    log_path = folder/"native.log"
    with log_path.open("x") as log:
        try:
            result = subprocess.run([native["openroad_executable"],"-no_init","-exit",str(script_path)],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("OpenROAD timed out on "+die) from exc
    text = log_path.read_text(errors="replace")
    marker = "MULTI_HBT_MATERIALIZED die=%s buffers=%d anchors=%d"%(die,buffers,anchors)
    if result.returncode or marker not in text or not odb.is_file():
        raise RuntimeError("OpenROAD %s materialization failed\n%s"%(die,"\n".join(text.splitlines()[-30:])))


def make_plan(context,layouts):
    physical = {}; targets = []; mapping = {}; suffix = context["new_hbt_id_base"]
    for target_index,logical in enumerate(sorted(context["target_nets"])):
        target,layout = context["target_nets"][logical],layouts[logical]
        rows = [dict(branch_id="multi_hbt_original_"+str(target_index),
            hbt_id=target["original_hbt_id"],site_id=target["original_site_id"],
            source_die=target["source_die"],sink_die=target["sink_die"],
            buffer_instance=None,receivers=layout.receivers[0],buffer=None)]
        direction = "B2T" if target["source_die"] == "bottom" else "T2B"
        for site,receiver_group,cell in zip(layout.sites,layout.receivers[1:],layout.buffers):
            rows.append(dict(branch_id="multi_hbt_branch_"+str(suffix),
                hbt_id="HBT_%s_%d"%(direction,suffix),site_id=site,
                source_die=target["source_die"],sink_die=target["sink_die"],
                buffer_instance="multi_hbt_iso_"+str(suffix),
                receivers=receiver_group,buffer=cell)); suffix += 1
        physical[logical] = rows
        public = [{k:r[k] for k in ("branch_id","hbt_id","site_id","source_die",
                  "sink_die","buffer_instance")} for r in rows]
        targets.append({"logical_net":logical,"branches":public})
        mapping[logical] = [r["branch_id"] for r in rows]
    return physical,{"branch_map":mapping,"target_nets":targets}


def synthesize(context,output_dir):
    """Select, materialize, legalize, and map one to three branches per target."""
    buffers = validate(context)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("output already exists: "+str(output))
    output.mkdir(parents=True)
    physical,plan = make_plan(context,choose(context,buffers))
    for die in DIES:
        run_native(context,output,die,physical)
    with (output/"branch_plan.json").open("x") as stream:
        json.dump(plan,stream,indent=2,sort_keys=True); stream.write("\n")
    return plan
