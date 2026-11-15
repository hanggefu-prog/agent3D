"""Adapt OpenROAD's direct-name estimated SPEF to the shared routed reader."""
from pathlib import Path
import re
import math


def validate_connectivity(source):
    """Every declared pin must belong to one connected, finite RC network."""
    name, section, pins, graph, count = None, None, [], {}, 0
    for line in Path(source).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "*D_NET":
            if name is not None:
                raise ValueError("Unterminated SPEF network")
            name, section, pins, graph = fields[1], None, [], {}
        elif fields[0] in {"*CONN", "*CAP", "*RES"}:
            section = fields[0]
        elif fields[0] == "*END" and name is not None:
            if not pins:
                raise ValueError("SPEF network has no pins: " + name)
            visited, pending = set(), [pins[0]]
            while pending:
                point = pending.pop()
                if point not in visited:
                    visited.add(point)
                    pending.extend(graph.get(point, set()) - visited)
            if any(pin not in visited for pin in pins):
                raise ValueError("Disconnected estimated RC pins: " + name)
            count += 1
            name, section = None, None
        elif name is not None and section == "*CONN" and fields[0] in {"*I", "*P"}:
            pins.append(fields[1])
        elif name is not None and section in {"*CAP", "*RES"} and fields[0].isdigit():
            value = float(fields[-1])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Nonfinite/negative estimated RC")
            if section == "*RES":
                if len(fields) != 4:
                    raise ValueError("Malformed SPEF resistor")
                a, b = fields[1:3]
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
    if name is not None or not count:
        raise ValueError("Incomplete estimated SPEF")
    return count


def add_name_map(source, destination, verilog=None):
    validate_connectivity(source)
    text = Path(source).read_text()
    if "*NAME_MAP" in text:
        raise ValueError("Expected the direct-name estimated SPEF writer")
    lines = text.splitlines()
    nets = [line.split()[1] for line in lines if line.startswith("*D_NET ")]
    instances = [line.split()[1].rsplit(":", 1)[0] for line in lines if line.startswith("*I ")]
    literal_names = set(instances)
    if verilog is not None:
        literal_names.update(re.findall(r"\\([^\s]+)", Path(verilog).read_text()))
    if not nets or len(set(nets)) != len(nets):
        raise ValueError("Missing or duplicate SPEF networks")
    names = {name: "*" + str(i + 1) for i, name in enumerate(sorted(set(nets + instances)))}

    def mapped_name(name):
        # Estimated SPEF omits escapes emitted by OpenRCX. Restore the same
        # distinction between literal flat names and genuine vector ports.
        pattern = r"(?<!\\)([./\[\]])" if name in literal_names else r"(?<!\\)([./])"
        return re.sub(pattern, r"\\\1", name)

    def node(value):
        if value in names:
            return names[value]
        if ":" in value:
            base, suffix = value.rsplit(":", 1)
            if base in names:
                return names[base] + ":" + suffix
        return value

    output, section, inserted = [], None, False
    for line in lines:
        fields = line.split()
        if fields and fields[0] in {"*PORTS", "*D_NET"} and not inserted:
            output += ["*NAME_MAP"] + [token + " " + mapped_name(name) for name, token in names.items()] + [""]
            inserted = True
        if not fields:
            output.append(line)
            continue
        if fields[0] == "*D_NET":
            fields[1] = names[fields[1]]
            section = None
        elif fields[0] in {"*CONN", "*CAP", "*RES", "*PORTS", "*END"}:
            section = fields[0]
        elif section == "*CONN" and fields[0] == "*I":
            fields[1] = node(fields[1])
        elif section in {"*CAP", "*RES"} and re.fullmatch(r"\d+", fields[0]):
            expected = {3, 4} if section == "*CAP" else {4}
            if len(fields) not in expected:
                raise ValueError("Malformed RC row")
            fields[1:-1] = [node(value) for value in fields[1:-1]]
        else:
            output.append(line)
            continue
        output.append(" ".join(fields))
    Path(destination).write_text("\n".join(output) + "\n")
