"""Independent buffer/inverter contraction proof for the CTS edit scope.

The input is protected native read-back, never candidate-produced JSON. Original
non-clock cells and their pin functions are unchanged. Liberty-proven clock
buffer/inverter paths may be reconstructed; their inversion parity and every
original logical endpoint must remain equivalent across the passive HBT links.
This is a scope-specific proof, not an arbitrary sequential-equivalence solver.
"""
import csv
from pathlib import Path

from .legalization import load_snapshot, restoration
from .protocol import digest


class ParitySets:
    def __init__(self):
        self.parent, self.parity, self.rank = {}, {}, {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x], self.parity[x], self.rank[x] = x, 0, 0
        if self.parent[x] != x:
            old = self.parent[x]
            root, parity = self.find(old)
            self.parent[x] = root
            self.parity[x] ^= parity
        return self.parent[x], self.parity[x]

    def join(self, a, b, inversion=0):
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            if pa ^ pb != inversion:
                raise ValueError("Contradictory inversion cycle")
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.parity[rb] = pa ^ pb ^ inversion
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def read_tsv(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if any(None in row for row in rows):
        raise ValueError("Malformed native pin/port table")
    return rows


def topology(snapshots, hbts, transparent_buffers=()):
    if set(snapshots) != {"bottom", "top"}:
        raise ValueError("Both die snapshots required")
    electrical, logical = ParitySets(), ParitySets()
    cell_tables, pin_tables, port_tables = {}, {}, {}
    boundary_ports = set()
    for die, folder in snapshots.items():
        _, _, cell_tables[die] = load_snapshot(folder)
        pin_tables[die] = read_tsv(Path(folder) / "pins.tsv")
        ports = {}
        for row in read_tsv(Path(folder) / "ports.tsv"):
            old = ports.get(row["port"])
            if old is not None and any(old[k] != row[k] for k in ("net", "direction", "signal_type")):
                raise ValueError("Inconsistent native port boxes")
            ports[row["port"]] = row
        port_tables[die] = ports
    seen_hbts = set()
    for row in hbts:
        if row["hbt_id"] in seen_hbts:
            raise ValueError("Repeated HBT identity")
        seen_hbts.add(row["hbt_id"])
        ends = []
        for side in ("source", "sink"):
            die, port = row[side + "_die"], row[side + "_port"]
            if die not in snapshots or (die, port) in boundary_ports:
                raise ValueError("Invalid/repeated HBT boundary")
            boundary_ports.add((die, port))
            pin = port_tables[die][port]
            if not pin["net"]:
                raise ValueError("Unconnected HBT port")
            ends.append((die, pin["net"]))
        electrical.join(*ends)
        logical.join(*ends)
    anchors, drivers, loads = {}, [], []
    for die, cells in cell_tables.items():
        pins_by_cell = {name: [] for name in cells}
        seen = set()
        for row in pin_tables[die]:
            key = (row["instance"], row["pin"])
            if key in seen:
                raise ValueError("Repeated native instance pin")
            seen.add(key)
            if row["instance"] not in pins_by_cell:
                raise ValueError("Pin points to missing cell")
            if row["signal_type"] in {"POWER", "GROUND"}:
                continue
            pins_by_cell[row["instance"]].append(row)
        for name, cell in cells.items():
            pins = pins_by_cell[name]
            if cell["master_type"].startswith("COVER"):
                continue  # Physical anchors; geometry and PG are separate gates.
            for pin in pins:
                if pin["net"]:
                    node = (die, pin["net"])
                    if pin["direction"] == "OUTPUT":
                        drivers.append((node, (die, name, pin["pin"])))
                    elif pin["direction"] == "INPUT":
                        loads.append(node)
                    else:
                        raise ValueError("Unsupported bidirectional signal pin")
            extra_buffer = (die, name) in transparent_buffers
            if extra_buffer and (not cell['buffer'] or cell['inverter']):
                raise ValueError('Added data primitive is not a native-proven noninverting buffer')
            if cell["clock_cell"] or extra_buffer:
                inputs = [p for p in pins if p["direction"] == "INPUT"]
                outputs = [p for p in pins if p["direction"] == "OUTPUT"]
                if len(pins) != 2 or len(inputs) != 1 or len(outputs) != 1 or not inputs[0]["net"]:
                    raise ValueError("Clock primitive is not a proven unary function")
                if outputs[0]["net"]:
                    logical.join((die, inputs[0]["net"]), (die, outputs[0]["net"]), cell["inverter"])
                elif not cell["cts_owned"]:
                    raise ValueError("Unowned dangling clock primitive")
            else:
                for pin in pins:
                    anchor = ("instance", die, name, cell["master"], pin["pin"], pin["direction"])
                    anchors[anchor] = (die, pin["net"]) if pin["net"] else ("unconnected", *anchor)
        for name, pin in port_tables[die].items():
            if (die, name) in boundary_ports or pin["signal_type"] in {"POWER", "GROUND"}:
                continue
            anchor = ("port", die, name, pin["direction"])
            node = (die, pin["net"]) if pin["net"] else ("unconnected", *anchor)
            anchors[anchor] = node
            if pin["net"]:
                if pin["direction"] == "INPUT":
                    drivers.append((node, anchor))
                elif pin["direction"] == "OUTPUT":
                    loads.append(node)
                else:
                    raise ValueError("Unsupported bidirectional external port")
    by_net = {}
    for node, driver in drivers:
        root, _ = electrical.find(node)
        by_net.setdefault(root, set()).add(driver)
    for node in loads:
        root, _ = electrical.find(node)
        if len(by_net.get(root, set())) != 1:
            raise ValueError("Logical load lacks exactly one electrical driver: " + str(node))
    groups = {}
    for anchor, node in anchors.items():
        root, phase = logical.find(node)
        groups.setdefault(root, []).append((anchor, phase))
    canonical = []
    for group in groups.values():
        ordered = sorted(group)
        offset = ordered[0][1]
        canonical.append(tuple((anchor, phase ^ offset) for anchor, phase in ordered))
    return sorted(canonical), len(anchors)


def compare(original, candidate, hbt_csv):
    for die in ("bottom", "top"):
        _, _, before = load_snapshot(original[die])
        _, _, after = load_snapshot(candidate[die])
        restoration(before, after)  # Preserve all original non-clock functions.
    with Path(hbt_csv).open() as stream:
        links = list(csv.DictReader(stream))
    first, old_count = topology(original, links)
    second, new_count = topology(candidate, links)
    if first != second:
        old_groups, new_groups = set(first), set(second)
        raise ValueError("Clock/data connectivity or inversion parity changed; differing groups="
                         + str(len(old_groups ^ new_groups)))
    return {"passed": True, "proof": "unchanged-data-functions-and-clock-buffer-inverter-contraction",
            "original_anchor_count": old_count, "candidate_anchor_count": new_count,
            "logical_connectivity_sha256": digest(first), "not_arbitrary_logic_equivalence": True}


def compare_local(original, candidate, original_hbts, candidate_hbts, added_buffers=(), added_hbts=()):
    """Fixed-original-cell ECO proof; only declared, read-back proven additions.

    Existing standard cells and clock trees cannot move, disappear or change
    masters. Added COVER anchors and noninverting buffer primitives are checked
    natively; their logical contraction must preserve every original endpoint.
    """
    added_buffers, added_hbts = set(added_buffers), set(added_hbts)
    for die in ('bottom', 'top'):
        _, _, before = load_snapshot(original[die])
        _, _, after = load_snapshot(candidate[die])
        for name, cell in before.items():
            if name not in after or any(cell[k] != after[name][k] for k in ('master', 'x_dbu', 'y_dbu', 'orient', 'status')):
                raise ValueError('Original local-task cell changed: ' + die + '/' + name)
        for name in set(after) - set(before):
            cell = after[name]
            if (die, name) in added_buffers:
                if not cell['buffer'] or cell['inverter']:
                    raise ValueError('Unproven added buffer function')
            elif (die, name) in added_hbts:
                if cell['master_type'] != 'COVER':
                    raise ValueError('New HBT is not a physical COVER')
            else:
                raise ValueError('Unmapped new physical instance')
        if any(d == die and n not in after for d, n in added_buffers | added_hbts):
            raise ValueError('Declared addition missing from native ODB')
    first, count = topology(original, original_hbts)
    second, new_count = topology(candidate, candidate_hbts, added_buffers)
    if first != second:
        raise ValueError('Local ECO changed original data/clock logical connectivity or polarity')
    return {'passed': True, 'proof': 'fixed-original-cells-and-native-noninverting-buffer-contraction',
            'original_anchor_count': count, 'candidate_anchor_count': new_count,
            'logical_connectivity_sha256': digest(first), 'not_arbitrary_logic_equivalence': True}
