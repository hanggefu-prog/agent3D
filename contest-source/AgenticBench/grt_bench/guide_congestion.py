"""Independent GCell-edge congestion estimate from frozen routing artifacts.

FastRoute prints its own final overflow table, and the post-GRT track scores
that printed number.  When the global router is candidate-editable, the
number is therefore self-reported by candidate code.  This module recomputes
an overflow figure without executing or importing any router code.  It reads
only artifacts the frozen flow writes:

* the route guide (``grt.guide``; written by OpenDB ``write_guides``)::

      net_name
      (
      xl yl xh yh layer
      ...
      )

* the die DEF (``global_route.odb.def``; written by OpenDB ``write_def``),
  from which DIEAREA, TRACKS and, informationally, GCELLGRID are taken;
* routing-layer preferred directions (tech LEF, an explicit mapping, or a
  declared built-in table for the maintained NG45/ASAP7 layer names);
* optionally the generated ``grt.tcl`` for the routing layer window and the
  ``set_global_routing_layer_adjustment`` derating.

Model (version 1)
-----------------
The GCell grid follows OpenDB's default: tile = 15 x the median preferred-
direction track pitch of routing levels 2..4 (or of the top routing level when
fewer than four are routed), anchored at the DIEAREA lower-left corner, with
``floor(extent / tile)`` GCells per axis and the last row/column absorbing the
remainder.  An explicit tile size can be supplied instead.

Capacity of a GCell edge on a layer is the number of that layer's preferred-
direction tracks inside the GCell row (horizontal layer) or column (vertical
layer), counted exactly like OpenROAD's nominal ``computeGCellCapacity``.  A
layer adjustment ``a`` gives ``max(1, floor(cap * (1 - a)))`` for ``cap > 0``
(FastRoute's single-precision rule).  Layers outside the routing window have
zero capacity.  Obstruction, macro, PDN and via-blockage reductions are NOT
modelled, so this capacity is an upper bound of FastRoute's.

Demand: every guide rectangle of a net on a layer marks the GCell edges it
spans along that layer's preferred direction.  Each net contributes at most
one unit per edge (overlapping rectangles of the same net are de-duplicated),
so NDR/wide-wire multipliers are not modelled and demand is a lower bound of
FastRoute's usage.  Single-GCell rectangles (vias, pin access) use no edge.

Overflow = sum over edges of ``max(0, demand - capacity)``; per-layer totals,
the maximum edge overflow and the number of overflowing edges are reported.

This is an independent ESTIMATE.  Its absolute value need not equal
FastRoute's internal overflow (FastRoute derates capacity for blockages and
charges wide wires more).  By construction it tends to be at most the true
FastRoute 3D overflow, which makes it useful as (a) an audit cross-check that
a router does not under-report overflow and (b) an alternative, independently
measured objective for NEW contracts after their references are re-qualified.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import re
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

ESTIMATOR_NAME = "guide-gcell-edge-overflow"
ESTIMATOR_VERSION = 1
PITCHES_IN_TILE = 15
UPPER_LAYER_FOR_GCELL_SIZE = 4
HORIZONTAL = "HORIZONTAL"
VERTICAL = "VERTICAL"
TRACK_MODELS = ("average", "exact")

# Declared fallback when no tech LEF is available.  Both tables were checked
# against the platform tech LEFs under OpenROAD-3D/flow/platforms.
BUILTIN_LAYER_DIRECTIONS = (
    ("builtin:nangate45-metalN", re.compile(r"metal(\d+)"),
     lambda index: HORIZONTAL if index % 2 else VERTICAL, {}),
    ("builtin:asap7-MN", re.compile(r"M(\d+)"),
     lambda index: VERTICAL if index % 2 else HORIZONTAL, {"Pad": HORIZONTAL}),
)

Rect = Tuple[int, int, int, int]


class GuideCongestionError(ValueError):
    """Raised when an input artifact is missing, malformed or ambiguous."""


def source_sha256() -> str:
    """Return the SHA-256 of this estimator's source file."""

    return _sha256(Path(__file__).resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _text(source: Union[str, Path]) -> str:
    if isinstance(source, Path):
        if not source.is_file():
            raise GuideCongestionError("missing input artifact: {}".format(source))
        return source.read_text(encoding="utf-8", errors="replace")
    return source


def _normalize_direction(value: str) -> str:
    text = str(value).strip().upper()
    if text in {"H", "HORIZONTAL"}:
        return HORIZONTAL
    if text in {"V", "VERTICAL"}:
        return VERTICAL
    raise GuideCongestionError("invalid layer direction: {!r}".format(value))


# --------------------------------------------------------------------------
# DEF
# --------------------------------------------------------------------------

_DEF_STOP = {"COMPONENTS", "PINS", "SPECIALNETS", "NETS", "BLOCKAGES", "FILLS", "GROUPS"}


def _def_statements(lines: Iterator[str]) -> Iterator[List[str]]:
    """Yield top-level header statements until the first big section.

    ``END <section>`` lines (no semicolon) of header sections such as
    PROPERTYDEFINITIONS or VIAS are skipped; ``END DESIGN`` stops.
    """

    tokens: List[str] = []
    skip_section_name = False
    for raw in lines:
        line = raw.split("#", 1)[0]
        for token in line.replace(";", " ; ").split():
            if skip_section_name:
                skip_section_name = False
                if token == "DESIGN":
                    return
                continue
            if not tokens and token in _DEF_STOP:
                return
            if not tokens and token == "END":
                skip_section_name = True
                continue
            if token == ";":
                if tokens:
                    yield tokens
                tokens = []
            else:
                tokens.append(token)
    if tokens:
        yield tokens


def parse_def_header(source: Union[str, Path]) -> Dict[str, object]:
    """Parse DIEAREA, UNITS, TRACKS and GCELLGRID from a DEF header.

    ``TRACKS X`` defines vertical track lines at x positions and ``TRACKS Y``
    defines horizontal track lines at y positions (DEF 5.8).  Patterns are
    returned as ``(start, count, step)`` tuples keyed by layer and axis.
    """

    if isinstance(source, Path):
        if not source.is_file():
            raise GuideCongestionError("missing DEF: {}".format(source))
        with source.open("r", encoding="utf-8", errors="replace") as stream:
            return _parse_def_statements(_def_statements(stream))
    return _parse_def_statements(_def_statements(iter(source.splitlines())))


def _int(token: str, label: str) -> int:
    try:
        return int(token)
    except ValueError:
        try:
            value = float(token)
        except ValueError:
            raise GuideCongestionError("invalid {} value: {!r}".format(label, token)) from None
        if not value.is_integer():
            raise GuideCongestionError("non-integral {} value: {!r}".format(label, token))
        return int(value)


def _parse_def_statements(statements: Iterator[List[str]]) -> Dict[str, object]:
    units: Optional[int] = None
    die: Optional[Rect] = None
    tracks: Dict[str, Dict[str, List[Tuple[int, int, int]]]] = {}
    gcell: Dict[str, List[Tuple[int, int, int]]] = {"X": [], "Y": []}
    for tokens in statements:
        keyword = tokens[0]
        if keyword == "UNITS" and len(tokens) >= 4 and tokens[1:3] == ["DISTANCE", "MICRONS"]:
            units = _int(tokens[3], "UNITS")
        elif keyword == "DIEAREA":
            numbers = [_int(token, "DIEAREA") for token in tokens[1:] if token not in {"(", ")"}]
            if len(numbers) < 4 or len(numbers) % 2:
                raise GuideCongestionError("malformed DIEAREA: {}".format(" ".join(tokens)))
            xs, ys = numbers[0::2], numbers[1::2]
            die = (min(xs), min(ys), max(xs), max(ys))
        elif keyword in {"TRACKS", "GCELLGRID"}:
            if len(tokens) < 6 or tokens[1] not in {"X", "Y"} or tokens[3] != "DO" or tokens[5] != "STEP":
                raise GuideCongestionError("malformed {}: {}".format(keyword, " ".join(tokens)))
            pattern = (_int(tokens[2], keyword), _int(tokens[4], keyword), _int(tokens[6], keyword))
            if pattern[1] < 0 or pattern[2] < 0:
                raise GuideCongestionError("negative {} pattern: {}".format(keyword, " ".join(tokens)))
            if keyword == "GCELLGRID":
                gcell[tokens[1]].append(pattern)
                continue
            if "LAYER" not in tokens:
                raise GuideCongestionError("TRACKS without LAYER: {}".format(" ".join(tokens)))
            layers = tokens[tokens.index("LAYER") + 1:]
            if not layers:
                raise GuideCongestionError("TRACKS without LAYER: {}".format(" ".join(tokens)))
            for layer in layers:
                tracks.setdefault(layer, {"X": [], "Y": []})[tokens[1]].append(pattern)
    if die is None:
        raise GuideCongestionError("DEF has no DIEAREA")
    if die[2] <= die[0] or die[3] <= die[1]:
        raise GuideCongestionError("DEF DIEAREA is empty: {}".format(die))
    return {
        "units_per_micron": units,
        "diearea": die,
        "tracks": tracks,
        "gcellgrid": gcell if (gcell["X"] or gcell["Y"]) else None,
    }


# --------------------------------------------------------------------------
# LEF / layer directions / layer order
# --------------------------------------------------------------------------

def parse_lef_layers(source: Union[str, Path]) -> Dict[str, object]:
    """Return routing-layer order and preferred directions from a tech LEF."""

    text = _text(source)
    order: List[str] = []
    directions: Dict[str, str] = {}
    current: Optional[str] = None
    routing = False
    direction: Optional[str] = None
    for raw in text.splitlines():
        tokens = raw.split("#", 1)[0].replace(";", " ; ").split()
        if not tokens:
            continue
        if len(tokens) == 2 and tokens[0] == "LAYER" and current is None:
            current, routing, direction = tokens[1], False, None
        elif current is not None and tokens[0] == "END" and len(tokens) >= 2 and tokens[1] == current:
            if routing and current not in directions:
                if direction is None:
                    raise GuideCongestionError("routing layer {} has no DIRECTION".format(current))
                order.append(current)
                directions[current] = direction
            current = None
        elif current is not None and tokens[0] == "TYPE" and len(tokens) >= 2:
            routing = tokens[1] == "ROUTING"
        elif current is not None and tokens[0] == "DIRECTION" and len(tokens) >= 2:
            direction = _normalize_direction(tokens[1])
    return {"order": order, "directions": directions}


def builtin_layer_directions(names: Sequence[str]) -> Tuple[Dict[str, str], str]:
    """Return a declared built-in direction table covering *names*, if any."""

    for label, pattern, rule, extra in BUILTIN_LAYER_DIRECTIONS:
        result: Dict[str, str] = {}
        for name in names:
            match = pattern.fullmatch(name)
            if match:
                result[name] = rule(int(match.group(1)))
            elif name in extra:
                result[name] = extra[name]
            else:
                break
        else:
            if result:
                return result, label
    raise GuideCongestionError(
        "no tech LEF or explicit directions, and layers {} match no built-in table".format(sorted(names))
    )


def infer_layer_order(names: Sequence[str]) -> List[str]:
    """Order layers by the trailing index of their dominant name family."""

    parsed = []
    for name in names:
        match = re.fullmatch(r"([A-Za-z_]+?)(\d+)", name)
        if match:
            parsed.append((match.group(1), int(match.group(2)), name))
    if not parsed:
        raise GuideCongestionError("cannot infer routing-layer order from {}".format(sorted(names)))
    families: Dict[str, int] = {}
    for family, _, _ in parsed:
        families[family] = families.get(family, 0) + 1
    family = max(sorted(families), key=lambda item: families[item])
    ordered = [name for fam, _, name in sorted(parsed, key=lambda row: row[1]) if fam == family]
    if len({index for fam, index, _ in parsed if fam == family}) != len(ordered):
        raise GuideCongestionError("duplicate routing-layer indices in {}".format(sorted(names)))
    return ordered + sorted(name for name in names if name not in ordered)


# --------------------------------------------------------------------------
# grt.tcl
# --------------------------------------------------------------------------

_ENV_ASSIGN = re.compile(r"^set ::env\((\w+)\) \{([^}]*)\}", re.MULTILINE)
_VAR_ASSIGN = re.compile(r"^\s*set\s+([A-Za-z_]\w*)\s+(\S+)\s*$")


def _substitute(text: str, env: Mapping[str, str], variables: Mapping[str, str]) -> str:
    text = re.sub(r"\$::env\((\w+)\)", lambda m: env.get(m.group(1), m.group(0)), text)
    text = re.sub(r"\$\{(\w+)\}", lambda m: variables.get(m.group(1), m.group(0)), text)
    return re.sub(r"\$(\w+)", lambda m: variables.get(m.group(1), m.group(0)), text)


def parse_grt_tcl(source: Union[str, Path]) -> Dict[str, object]:
    """Extract the routing window and layer adjustments from a GRT script.

    Only simple, fully resolvable assignments are evaluated: ``set ::env(K)
    {V}``, ``set name value`` (with ``$::env``/``$var`` substitution), the
    ``set_routing_layers -signal`` window and every
    ``set_global_routing_layer_adjustment LAYER|RANGE|* VALUE`` command.
    """

    text = _text(source)
    env = {key: value for key, value in _ENV_ASSIGN.findall(text)}
    variables: Dict[str, str] = {}
    layer_range: Optional[Tuple[str, str]] = None
    adjustments: List[Tuple[str, float]] = []
    # Environment values may span lines; drop them before scanning commands.
    for raw in _ENV_ASSIGN.sub("", text).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _VAR_ASSIGN.match(line)
        if match and match.group(1) != "::env":
            value = _substitute(match.group(2), env, variables)
            if "$" not in value and "[" not in value:
                variables[match.group(1)] = value
            continue
        tokens = _substitute(line, env, variables).split()
        if tokens[0] == "set_routing_layers" and "-signal" in tokens:
            spec = tokens[tokens.index("-signal") + 1]
            if spec.count("-") != 1 or "$" in spec:
                raise GuideCongestionError("unresolved routing layer window: {}".format(spec))
            low, high = spec.split("-")
            layer_range = (low, high)
        elif tokens[0] == "set_global_routing_layer_adjustment":
            if len(tokens) != 3 or any("$" in token or "[" in token for token in tokens):
                raise GuideCongestionError("unresolved layer adjustment: {}".format(raw.strip()))
            try:
                value = float(tokens[2])
            except ValueError:
                raise GuideCongestionError("invalid layer adjustment: {}".format(raw.strip())) from None
            adjustments.append((tokens[1], value))
    return {"env": env, "layer_range": layer_range, "adjustments": adjustments}


def expand_adjustments(
    adjustments: Sequence[Tuple[str, float]], order: Sequence[str]
) -> Dict[str, float]:
    """Expand ``(spec, value)`` pairs (``*``, ``a-b`` or a layer) per layer.

    As in OpenROAD, a per-layer value set later overrides an earlier one, and
    the global ``*`` value applies only to layers without a layer value.
    """

    per_layer: Dict[str, float] = {}
    global_value: Optional[float] = None
    for spec, value in adjustments:
        if not 0.0 <= value <= 1.0:
            raise GuideCongestionError("layer adjustment outside [0, 1]: {} {}".format(spec, value))
        if spec == "*":
            global_value = value
            continue
        if "-" in spec:
            low, high = spec.split("-", 1)
            if low not in order or high not in order:
                raise GuideCongestionError("unknown adjustment layer range: {}".format(spec))
            names = order[order.index(low):order.index(high) + 1]
        else:
            if spec not in order:
                raise GuideCongestionError("unknown adjustment layer: {}".format(spec))
            names = [spec]
        for name in names:
            per_layer[name] = value
    if global_value:
        for name in order:
            if not per_layer.get(name):
                per_layer[name] = global_value
    return per_layer


# --------------------------------------------------------------------------
# Guides
# --------------------------------------------------------------------------

def iter_guides(source: Union[str, Path]) -> Iterator[Tuple[str, List[Tuple[int, int, int, int, str]]]]:
    """Yield ``(net, [(xl, yl, xh, yh, layer), ...])`` blocks in file order.

    Blank lines and ``#`` comments are ignored; anything else that does not
    follow the ``write_guides`` grammar raises :class:`GuideCongestionError`.
    """

    if isinstance(source, Path):
        if not source.is_file():
            raise GuideCongestionError("missing guide file: {}".format(source))
        with source.open("r", encoding="utf-8", errors="replace") as stream:
            for item in _iter_guide_lines(stream, str(source)):
                yield item
    else:
        for item in _iter_guide_lines(iter(source.splitlines()), "<text>"):
            yield item


def _iter_guide_lines(lines, label):
    net: Optional[str] = None
    rects: Optional[List[Tuple[int, int, int, int, str]]] = None
    for number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if rects is None:
            if net is None:
                if line in {"(", ")"} or len(line.split()) != 1:
                    raise GuideCongestionError("{}:{}: expected a net name".format(label, number))
                net = line
            elif line == "(":
                rects = []
            else:
                raise GuideCongestionError("{}:{}: expected '(' after net {}".format(label, number, net))
            continue
        if line == ")":
            yield net, rects
            net, rects = None, None
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise GuideCongestionError("{}:{}: malformed guide line {!r}".format(label, number, line))
        try:
            xl, yl, xh, yh = (int(token) for token in tokens[:4])
        except ValueError:
            raise GuideCongestionError("{}:{}: non-integer guide coordinates".format(label, number)) from None
        if xh < xl or yh < yl:
            raise GuideCongestionError("{}:{}: inverted guide rectangle".format(label, number))
        rects.append((xl, yl, xh, yh, tokens[4]))
    if net is not None:
        raise GuideCongestionError("{}: truncated guide block for net {}".format(label, net))


# --------------------------------------------------------------------------
# Grid and capacity
# --------------------------------------------------------------------------

def _pattern_positions(patterns: Sequence[Tuple[int, int, int]]) -> List[int]:
    positions = set()
    for start, count, step in patterns:
        if count > 1 and step <= 0:
            raise GuideCongestionError("non-positive track step")
        positions.update(start + index * step for index in range(count))
    return sorted(positions)


def preferred_tracks(
    header: Mapping[str, object], layer: str, direction: str, model: str = "average"
) -> Dict[str, object]:
    """Return a layer's preferred-direction track model.

    ``average`` reproduces OpenDB ``getAverageTrackSpacing`` (single pattern
    verbatim; several patterns -> first coordinate, ``ceil(span / n)`` step,
    ``n`` tracks).  ``exact`` keeps the exact union of track coordinates.
    """

    if model not in TRACK_MODELS:
        raise GuideCongestionError("unknown track model: {}".format(model))
    grids = header["tracks"].get(layer)  # type: ignore[union-attr]
    axis = "Y" if direction == HORIZONTAL else "X"
    patterns = (grids or {}).get(axis) or []
    if not patterns:
        return {"init": 0, "step": 0, "count": 0, "positions": []}
    if model == "exact":
        positions = _pattern_positions(patterns)
        step = patterns[0][2] if len(patterns) == 1 else int(
            math.ceil(_f32(_f32(positions[-1] - positions[0]) / len(positions)))
        )
        return {"init": positions[0], "step": step, "count": len(positions), "positions": positions}
    if len(patterns) == 1:
        start, count, step = patterns[0]
        return {"init": start, "step": step, "count": count, "positions": None}
    positions = _pattern_positions(patterns)
    span = positions[-1] - positions[0]
    step = int(math.ceil(_f32(_f32(span) / len(positions))))
    return {"init": positions[0], "step": step, "count": len(positions), "positions": None}


def default_gcell_size(
    header: Mapping[str, object],
    order: Sequence[str],
    directions: Mapping[str, str],
    max_routing_level: Optional[int] = None,
) -> int:
    """Reproduce OpenDB ``dbBlock::getGCellTileSize`` from DEF tracks."""

    def spacing(level: int) -> int:
        if not 1 <= level <= len(order):
            raise GuideCongestionError("routing level {} is not defined".format(level))
        name = order[level - 1]
        if name not in directions:
            raise GuideCongestionError("no preferred direction for {}".format(name))
        tracks = preferred_tracks(header, name, directions[name], "average")
        if not tracks["count"] or int(tracks["step"]) <= 0:
            raise GuideCongestionError("no preferred-direction tracks for {}".format(name))
        step = int(tracks["step"])
        return step if step % 2 == 0 else step - 1

    top = max_routing_level if max_routing_level is not None else len(order)
    if top < UPPER_LAYER_FOR_GCELL_SIZE:
        return spacing(top) * PITCHES_IN_TILE
    values = sorted(spacing(level) for level in (2, 3, 4))
    return values[1] * PITCHES_IN_TILE


def _count_tracks(tracks: Mapping[str, object], low: int, high: int) -> int:
    """Count tracks with coordinate in ``[low, high)`` (computeGCellCapacity)."""

    positions = tracks.get("positions")
    if positions is not None:
        return bisect.bisect_left(positions, high) - bisect.bisect_left(positions, low)  # type: ignore[arg-type]
    init, pitch, count = int(tracks["init"]), int(tracks["step"]), int(tracks["count"])
    if count <= 0:
        return 0
    if pitch <= 0:
        return 1 if count and low <= init < high else 0
    first = 0 if low <= init else (low - init + pitch - 1) // pitch
    last = -1 if high < init else (high - init - 1) // pitch
    first = max(0, first)
    last = min(count - 1, last)
    return 0 if first > last else last - first + 1


def adjusted_capacity(capacity: int, adjustment: float) -> int:
    """FastRoute ``computeUserLayerAdjustments`` in single precision."""

    if adjustment == 0:
        return capacity
    factor = _f32(1.0 - _f32(adjustment))
    value = int(math.floor(_f32(_f32(float(capacity)) * factor)))
    if capacity > 0 and adjustment != 1:
        value = max(value, 1)
    return value


def _bounds(origin: int, tile: int, cells: int, extent_high: int) -> List[Tuple[int, int]]:
    rows = []
    for index in range(cells):
        low = origin + index * tile
        high = extent_high if index == cells - 1 else low + tile
        rows.append((low, high))
    return rows


# --------------------------------------------------------------------------
# Estimate
# --------------------------------------------------------------------------

def estimate_overflow(
    def_path: Union[str, Path],
    guide_path: Union[str, Path],
    *,
    layer_directions: Optional[Mapping[str, str]] = None,
    layer_order: Optional[Sequence[str]] = None,
    direction_source: Optional[str] = None,
    gcell_size: Optional[int] = None,
    layer_range: Optional[Tuple[str, str]] = None,
    layer_adjustments: Optional[Mapping[str, float]] = None,
    track_model: str = "average",
) -> Dict[str, object]:
    """Estimate per-layer GCell-edge overflow from a DEF and a route guide.

    ``def_path``/``guide_path`` may be :class:`~pathlib.Path` objects or raw
    text.  ``layer_order`` lists routing layers bottom-up (routing level 1
    first); it is inferred from layer names when omitted.  Without a
    ``layer_range`` every layer with tracks has capacity.
    """

    header = parse_def_header(Path(def_path) if isinstance(def_path, Path) else def_path)
    track_layers = sorted(header["tracks"])  # type: ignore[arg-type]
    order = list(layer_order) if layer_order else infer_layer_order(track_layers)
    if layer_directions:
        directions = {name: _normalize_direction(value) for name, value in layer_directions.items()}
        source = direction_source or "explicit"
    else:
        directions, source = builtin_layer_directions(sorted(set(order) | set(track_layers)))
    level = {name: index + 1 for index, name in enumerate(order)}
    if layer_range is not None:
        for name in layer_range:
            if name not in level:
                raise GuideCongestionError("routing window layer {} is not a known routing layer".format(name))
        low_level, high_level = level[layer_range[0]], level[layer_range[1]]
        if high_level < low_level:
            raise GuideCongestionError("inverted routing window {}".format(layer_range))
    else:
        low_level, high_level = 1, len(order)
    adjustments = dict(layer_adjustments or {})
    for name, value in adjustments.items():
        if not 0.0 <= float(value) <= 1.0:
            raise GuideCongestionError("layer adjustment outside [0, 1]: {}={}".format(name, value))

    xl, yl, xh, yh = header["diearea"]  # type: ignore[misc]
    if gcell_size is not None:
        tile = int(gcell_size)
        grid_source = "explicit"
    else:
        tile = default_gcell_size(header, order, directions, high_level if layer_range else None)
        grid_source = "derived-odb-default"
    if tile <= 0:
        raise GuideCongestionError("GCell size must be positive")
    nx = max(1, (xh - xl) // tile)
    ny = max(1, (yh - yl) // tile)
    columns = _bounds(xl, tile, nx, xh)
    rows = _bounds(yl, tile, ny, yh)

    def in_range(name: str) -> bool:
        return name in level and low_level <= level[name] <= high_level

    capacity_cache: Dict[str, List[int]] = {}

    def capacities(name: str) -> List[int]:
        if name not in capacity_cache:
            direction = directions[name]
            spans = rows if direction == HORIZONTAL else columns
            if not in_range(name):
                capacity_cache[name] = [0] * len(spans)
            else:
                tracks = preferred_tracks(header, name, direction, track_model)
                adjustment = float(adjustments.get(name, 0.0))
                capacity_cache[name] = [
                    adjusted_capacity(_count_tracks(tracks, low, high), adjustment) for low, high in spans
                ]
        return capacity_cache[name]

    def gcell_index(value: int, origin: int, cells: int) -> int:
        return min(cells - 1, max(0, (value - origin) // tile))

    demand: Dict[str, List[int]] = {}
    nets = rects_total = ignored_outside = local_rects = nonpreferred = 0
    pending: Dict[str, Dict[str, set]] = {}

    def edges_for(net_edges: Dict[str, set], rect: Tuple[int, int, int, int, str]) -> None:
        nonlocal ignored_outside, local_rects, nonpreferred
        rxl, ryl, rxh, ryh, layer = rect
        if rxh < xl or rxl > xh or ryh < yl or ryl > yh:
            ignored_outside += 1
            return
        c0 = gcell_index(rxl, xl, nx)
        c1 = gcell_index(max(rxl, rxh - 1), xl, nx)
        r0 = gcell_index(ryl, yl, ny)
        r1 = gcell_index(max(ryl, ryh - 1), yl, ny)
        if c0 == c1 and r0 == r1:
            local_rects += 1
            return
        if layer not in directions:
            raise GuideCongestionError("guide uses layer {} with unknown preferred direction".format(layer))
        horizontal = directions[layer] == HORIZONTAL
        if (r0 != r1) if horizontal else (c0 != c1):
            # A box wider than one GCell across the preferred direction is
            # unusual for FastRoute guides; it is counted, and each of its
            # rows/columns is charged along the preferred direction only.
            nonpreferred += 1
        target = net_edges.setdefault(layer, set())
        if horizontal:
            for row in range(r0, r1 + 1):
                base = row * (nx - 1)
                target.update(range(base + c0, base + c1))
        else:
            for column in range(c0, c1 + 1):
                base = column * (ny - 1)
                target.update(range(base + r0, base + r1))

    def commit(net_edges: Dict[str, set]) -> None:
        for layer, indices in net_edges.items():
            if layer not in demand:
                size = ny * (nx - 1) if directions[layer] == HORIZONTAL else nx * (ny - 1)
                demand[layer] = [0] * size
            values = demand[layer]
            for index in indices:
                values[index] += 1

    guide_source = Path(guide_path) if isinstance(guide_path, Path) else guide_path
    # Pass one counts blocks per net: write_guides emits each net once, but a
    # repeated block is merged so a net still adds at most one unit per edge.
    remaining: Dict[str, int] = {}
    for net, _ in iter_guides(guide_source):
        remaining[net] = remaining.get(net, 0) + 1
    duplicates = sum(count - 1 for count in remaining.values())
    for net, rects in iter_guides(guide_source):
        rects_total += len(rects)
        net_edges = pending.setdefault(net, {})
        for rect in rects:
            edges_for(net_edges, rect)
        remaining[net] -= 1
        if remaining[net] == 0:
            commit(pending.pop(net))
            nets += 1
    if pending:
        raise GuideCongestionError("internal guide accounting error")

    layers_report: Dict[str, object] = {}
    total = max_overflow = overflowing = capacity_total = demand_total = 0
    report_layers = [name for name in order if name in header["tracks"] or name in demand]  # type: ignore[operator]
    report_layers += sorted(name for name in demand if name not in report_layers)
    for name in report_layers:
        if name not in directions:
            continue
        direction = directions[name]
        caps = capacities(name)
        values = demand.get(name)
        if direction == HORIZONTAL:
            edge_count, lanes = ny * (nx - 1), nx - 1
        else:
            edge_count, lanes = nx * (ny - 1), ny - 1
        layer_capacity = sum(cap * lanes for cap in caps)
        layer_demand = layer_overflow = layer_max = layer_edges = 0
        if values:
            for index, used in enumerate(values):
                if not used:
                    continue
                layer_demand += used
                cap = caps[index // lanes] if lanes else 0
                excess = used - cap
                if excess > 0:
                    layer_overflow += excess
                    layer_edges += 1
                    layer_max = max(layer_max, excess)
        layers_report[name] = {
            "direction": direction,
            "in_routing_range": in_range(name),
            "adjustment": float(adjustments.get(name, 0.0)) if in_range(name) else None,
            "edges": edge_count,
            "capacity": layer_capacity,
            "demand": layer_demand,
            "overflow": layer_overflow,
            "max_edge_overflow": layer_max,
            "overflowing_edges": layer_edges,
        }
        total += layer_overflow
        max_overflow = max(max_overflow, layer_max)
        overflowing += layer_edges
        capacity_total += layer_capacity
        demand_total += layer_demand

    def_gcell = header["gcellgrid"]
    agrees: Optional[bool] = None
    if def_gcell:
        # GCELLGRID is written by the (possibly candidate) router, so it is
        # only compared, never used.  DO may count GCells or grid lines.
        def matches(axis: str, origin: int, cells: int) -> bool:
            patterns = def_gcell[axis]  # type: ignore[index]
            return len(patterns) == 1 and patterns[0][0] == origin and patterns[0][2] == tile \
                and patterns[0][1] in {cells, cells + 1}

        agrees = matches("X", xl, nx) and matches("Y", yl, ny)
    return {
        "schema_version": 1,
        "estimator": ESTIMATOR_NAME,
        "estimator_version": ESTIMATOR_VERSION,
        "track_model": track_model,
        "grid": {
            "origin": [xl, yl],
            "diearea": [xl, yl, xh, yh],
            "gcell_size": tile,
            "x_gcells": nx,
            "y_gcells": ny,
            "source": grid_source,
            "def_gcellgrid": (
                {axis: [list(item) for item in def_gcell[axis]] for axis in ("X", "Y")}  # type: ignore[index]
                if def_gcell else None
            ),
            "def_gcellgrid_agrees": agrees,
        },
        "direction_source": source,
        "layer_order": order,
        "layer_range": list(layer_range) if layer_range else None,
        "layers": layers_report,
        "total_overflow": total,
        "max_edge_overflow": max_overflow,
        "overflowing_edges": overflowing,
        "capacity": capacity_total,
        "demand": demand_total,
        "nets": nets,
        "rects": rects_total,
        "single_gcell_rects": local_rects,
        "nonpreferred_rects": nonpreferred,
        "rects_outside_die": ignored_outside,
        "duplicate_net_blocks": duplicates,
    }


def find_tech_lef(platform: Path) -> Optional[Path]:
    """Return the maintained tech LEF under an Open3DBench platform."""

    preferred = platform / "lef" / "NangateOpenCellLibrary.tech.lef"
    if preferred.is_file():
        return preferred
    candidates = sorted((platform / "lef").glob("*tech*.lef")) if (platform / "lef").is_dir() else []
    return candidates[0] if candidates else None


def estimate_die(
    directory: Path,
    *,
    tech_lef: Optional[Path] = None,
    platform: Optional[Path] = None,
    layer_directions: Optional[Mapping[str, str]] = None,
    gcell_size: Optional[int] = None,
    track_model: str = "average",
    use_grt_tcl: bool = True,
) -> Dict[str, object]:
    """Estimate one die from a post-GRT directory written by three_d_route.

    Reads ``global_route.odb.def``, ``grt.guide`` and (when ``use_grt_tcl``)
    ``grt.tcl``.  The tech LEF is ``tech_lef``, else found under
    ``platform``, else under the platform recorded by ``grt.tcl``
    (``ROUTE_SET_RC``'s directory); without one the built-in direction table
    is used and declared in ``direction_source``.
    """

    directory = Path(directory)
    def_path = directory / "global_route.odb.def"
    guide_path = directory / "grt.guide"
    for path in (def_path, guide_path):
        if not path.is_file() or path.is_symlink():
            raise GuideCongestionError("missing or symlinked artifact: {}".format(path))
    settings: Dict[str, object] = {"env": {}, "layer_range": None, "adjustments": []}
    if use_grt_tcl:
        script = directory / "grt.tcl"
        if not script.is_file() or script.is_symlink():
            raise GuideCongestionError("missing or symlinked artifact: {}".format(script))
        settings = parse_grt_tcl(script)
    lef = Path(tech_lef) if tech_lef else None
    if lef is None and platform is not None:
        lef = find_tech_lef(Path(platform))
    recorded_rc = settings["env"].get("ROUTE_SET_RC") if settings["env"] else None  # type: ignore[union-attr]
    if lef is None and recorded_rc:
        lef = find_tech_lef(Path(str(recorded_rc)).parent)
    order: Optional[List[str]] = None
    directions = dict(layer_directions) if layer_directions else None
    source = "explicit" if directions else None
    if lef is not None:
        if not lef.is_file():
            raise GuideCongestionError("missing tech LEF: {}".format(lef))
        parsed = parse_lef_layers(lef)
        order = list(parsed["order"])  # type: ignore[arg-type]
        if directions is None:
            directions = dict(parsed["directions"])  # type: ignore[arg-type]
            source = "lef:{}:{}".format(lef.name, _sha256(lef))
    if order is None:
        header_layers = sorted(parse_def_header(def_path)["tracks"])  # type: ignore[arg-type]
        order = infer_layer_order(header_layers)
    adjustments = expand_adjustments(settings["adjustments"], order)  # type: ignore[arg-type]
    result = estimate_overflow(
        def_path,
        guide_path,
        layer_directions=directions,
        layer_order=order,
        direction_source=source,
        gcell_size=gcell_size,
        layer_range=settings["layer_range"],  # type: ignore[arg-type]
        layer_adjustments=adjustments,
        track_model=track_model,
    )
    result["inputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in ([def_path, guide_path] + ([directory / "grt.tcl"] if use_grt_tcl else []))
    }
    result["tech_lef"] = str(lef) if lef is not None else None
    return result
