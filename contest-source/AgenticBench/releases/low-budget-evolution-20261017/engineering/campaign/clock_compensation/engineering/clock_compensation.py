"""Select native post-CTS clock-branch compensation actions.

The protected executor owns database edits, legalization, and final STA.  This
module only chooses opaque action IDs.  Liberty estimates are action-local
predictions, not measurements of a hypothetical edited tree.
"""

from dataclasses import dataclass
import math


_SCHEMA = "clock-compensation-context-v2"
_CHAIN_LENGTHS = frozenset((1, 2, 4, 8))
_LOW_DELAY_FACTOR = 0.90
_HIGH_DELAY_FACTOR = 1.10
_PAIR_ACTION_LIMIT = 32
_MINIMUM_GAIN_NS = 0.001


@dataclass(frozen=True)
class _Group:
    name: str
    identity: tuple
    analysis: str
    arrivals: dict
    span: float


@dataclass(frozen=True)
class _Action:
    action_id: str
    branch_id: str
    target_input_pin: str
    sinks: frozenset
    sites: frozenset
    buffers: int
    area: float
    delay_min: float
    delay_max: float
    identity: tuple

    def delay(self, analysis):
        return self.delay_max if analysis == "max" else self.delay_min


@dataclass(frozen=True)
class _Prediction:
    actions: tuple
    worst_span: float
    total_span: float
    distribution_gain: float
    buffers: int
    area: float

    @property
    def ids(self):
        return tuple(sorted(action.action_id for action in self.actions))


def _number(value):
    # bool is an int in Python, but is not a physical number here.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _groups(context):
    rows = context.get("clock_groups")
    if not isinstance(rows, dict) or not rows:
        return None
    result = []
    analyses_by_identity = {}
    for name in sorted(rows):
        arrivals = rows[name]
        if not isinstance(name, str) or not isinstance(arrivals, dict) or not arrivals:
            return None
        fields = name.rsplit("/", 3)
        if len(fields) != 4:
            return None
        clock, edge, analysis, scenario = fields
        if (
            not clock
            or edge not in ("rise", "fall")
            or analysis not in ("max", "min")
            or not scenario
        ):
            return None
        checked = {}
        for sink, raw_arrival in arrivals.items():
            arrival = _number(raw_arrival)
            if not isinstance(sink, str) or not sink or arrival is None:
                return None
            checked[sink] = arrival
        identity = (clock, edge, scenario)
        analyses_by_identity.setdefault(identity, set()).add(analysis)
        result.append(
            _Group(
                name,
                identity,
                analysis,
                checked,
                max(checked.values()) - min(checked.values()),
            )
        )
    # Neither missing max data nor missing min data means zero delay.
    if any(value != {"max", "min"} for value in analyses_by_identity.values()):
        return None
    return result


def _timing(context):
    timing = context.get("timing")
    tolerances = context.get("timing_tolerances")
    paths = context.get("paths")
    if (
        not isinstance(timing, dict)
        or not isinstance(tolerances, dict)
        or not isinstance(paths, list)
        or not paths
    ):
        return None

    checked_tolerances = {}
    for key in ("setup_wns_ns", "setup_tns_ns", "hold_wns_ns", "hold_tns_ns"):
        metric = _number(timing.get(key))
        tolerance = _number(tolerances.get(key))
        if metric is None or tolerance is None or tolerance < 0.0:
            return None
        checked_tolerances[key] = tolerance

    violators = timing.get("hold_violating_endpoints")
    if not isinstance(violators, list) or any(
        not isinstance(endpoint, str) or not endpoint for endpoint in violators
    ):
        return None

    analyses = set()
    for path in paths:
        if not isinstance(path, dict):
            return None
        analysis = path.get("analysis")
        if (
            analysis not in ("max", "min")
            or _number(path.get("slack_ns")) is None
            or not isinstance(path.get("endpoint"), str)
            or not path.get("endpoint")
            or not isinstance(path.get("launch_clock_sink"), str)
            or not isinstance(path.get("capture_clock_sink"), str)
        ):
            return None
        analyses.add(analysis)
    if analyses != {"max", "min"}:
        return None
    return paths, frozenset(violators), checked_tolerances


def _limits(context):
    limits = context.get("limits")
    if not isinstance(limits, dict):
        return None
    buffers = limits.get("max_added_buffers")
    area = _number(limits.get("max_added_area_um2"))
    if isinstance(buffers, bool) or not isinstance(buffers, int):
        return None
    if buffers < 0 or area is None or area < 0.0:
        return None
    return buffers, area


def _actions(context, groups):
    rows = context.get("actions")
    cells = context.get("buffer_cells")
    if not isinstance(rows, dict) or not isinstance(cells, dict):
        return []

    sink_identities = {}
    for group in groups:
        for sink in group.arrivals:
            sink_identities.setdefault(sink, set()).add(group.identity)

    result = []
    for action_id in sorted(rows):
        row = rows[action_id]
        if not isinstance(action_id, str) or not isinstance(row, dict):
            continue
        if row.get("qualified") is not True or row.get("exclusive") is not True:
            continue
        branch = row.get("branch_id")
        target = row.get("target_input_pin")
        master = row.get("master")
        kind = row.get("estimate_kind")
        buffers = row.get("added_buffers")
        area = _number(row.get("added_area_um2"))
        slew = _number(row.get("input_slew_ns"))
        load = _number(row.get("load_pf"))
        if not all(
            isinstance(value, str) and value
            for value in (action_id, branch, target, master, kind)
        ):
            continue
        if master not in cells or isinstance(buffers, bool) or buffers not in _CHAIN_LENGTHS:
            continue
        if (
            area is None
            or area <= 0.0
            or slew is None
            or slew < 0.0
            or load is None
            or load < 0.0
        ):
            continue

        raw_sinks = row.get("affected_original_sinks")
        raw_sites = row.get("placement_site_ids")
        if not isinstance(raw_sinks, list) or not isinstance(raw_sites, list):
            continue
        if any(
            not isinstance(value, str) or not value
            for value in raw_sinks + raw_sites
        ):
            continue
        sinks = frozenset(raw_sinks)
        sites = frozenset(raw_sites)
        if not sinks or len(sinks) != len(raw_sinks):
            continue
        if len(sites) != buffers or len(sites) != len(raw_sites):
            continue

        delay = row.get("delay_estimate_ns")
        if not isinstance(delay, dict):
            continue
        delay_min = _number(delay.get("min"))
        delay_max = _number(delay.get("max"))
        if (
            delay_min is None
            or delay_max is None
            or delay_min <= 0.0
            or delay_max <= 0.0
        ):
            continue

        identities = set()
        for sink in sinks:
            known = sink_identities.get(sink)
            if not known or len(known) != 1:
                identities = set()
                break
            identities.update(known)
        if len(identities) != 1:
            continue
        result.append(
            _Action(
                action_id,
                branch,
                target,
                sinks,
                sites,
                buffers,
                area,
                delay_min,
                delay_max,
                next(iter(identities)),
            )
        )
    return result


def _compatible(actions, max_buffers, max_area):
    branches, targets, sinks, sites = set(), set(), set(), set()
    count, area = 0, 0.0
    for action in actions:
        if action.branch_id in branches or action.target_input_pin in targets:
            return False
        if not sinks.isdisjoint(action.sinks) or not sites.isdisjoint(action.sites):
            return False
        branches.add(action.branch_id)
        targets.add(action.target_input_pin)
        sinks.update(action.sinks)
        sites.update(action.sites)
        count += action.buffers
        area += action.area
    return count <= max_buffers and area <= max_area + 1.0e-12


def _predict(actions, groups, max_buffers, max_area):
    if not actions or not _compatible(actions, max_buffers, max_area):
        return None
    by_sink = {
        sink: action for action in actions for sink in action.sinks
    }
    spans = []
    distribution_gain = 0.0
    for group in groups:
        shifted = []
        midpoint = min(group.arrivals.values()) + group.span / 2.0
        for sink, arrival in group.arrivals.items():
            action = by_sink.get(sink)
            delta = 0.0
            if action is not None:
                if action.identity != group.identity:
                    return None
                delta = _LOW_DELAY_FACTOR * action.delay(group.analysis)
                distribution_gain += max(
                    0.0,
                    min(midpoint, arrival + delta) - min(midpoint, arrival),
                )
            shifted.append(arrival + delta)
        span = max(shifted) - min(shifted)
        # Do not improve one analysis view by worsening another.
        if span > group.span + 1.0e-12:
            return None
        spans.append(span)
    return _Prediction(
        tuple(actions),
        max(spans),
        sum(spans),
        distribution_gain,
        sum(action.buffers for action in actions),
        sum(action.area for action in actions),
    )


def _path_delta(path, by_sink):
    analysis = path["analysis"]
    launch_name = path["launch_clock_sink"]
    capture_name = path["capture_clock_sink"]
    launch = by_sink.get(launch_name) if launch_name else None
    capture = by_sink.get(capture_name) if capture_name else None

    # A common branch shifts both clocks together and earns no useful-skew
    # credit.  It is one edit, so its common component cancels.
    if launch is not None and launch is capture:
        return 0.0

    both_known = bool(launch_name and capture_name)
    delta = 0.0
    if analysis == "max":
        if launch is not None:
            delta -= _HIGH_DELAY_FACTOR * launch.delay_max
        if capture is not None and both_known:
            delta += _LOW_DELAY_FACTOR * capture.delay_max
    else:
        if capture is not None:
            delta -= _HIGH_DELAY_FACTOR * capture.delay_min
        if launch is not None and both_known:
            delta += _LOW_DELAY_FACTOR * launch.delay_min
    return delta


def _timing_safe(prediction, paths, hold_violators, tolerances):
    by_sink = {
        sink: action
        for action in prediction.actions
        for sink in action.sinks
    }
    old = {"max": [], "min": []}
    new = {"max": [], "min": []}
    for path in paths:
        analysis = path["analysis"]
        old_slack = float(path["slack_ns"])
        new_slack = old_slack + _path_delta(path, by_sink)
        old[analysis].append(old_slack)
        new[analysis].append(new_slack)
        if (
            analysis == "min"
            and path["endpoint"] not in hold_violators
            and new_slack < -tolerances["hold_wns_ns"]
        ):
            return False

    for analysis, prefix in (("max", "setup"), ("min", "hold")):
        if min(new[analysis]) < min(old[analysis]) - tolerances[prefix + "_wns_ns"]:
            return False
        old_tns = sum(value for value in old[analysis] if value < 0.0)
        new_tns = sum(value for value in new[analysis] if value < 0.0)
        if new_tns < old_tns - tolerances[prefix + "_tns_ns"]:
            return False
    return True


def _candidates(actions, groups, max_buffers, max_area, minimum_gain):
    baseline = max(group.span for group in groups)
    singles, pair_pool = [], []
    for action in actions:
        prediction = _predict((action,), groups, max_buffers, max_area)
        if prediction is None or prediction.distribution_gain <= 0.0:
            continue
        pair_pool.append(prediction)
        if baseline - prediction.worst_span >= minimum_gain:
            singles.append(prediction)

    # Preserve branch diversity for pair look-ahead: one efficient and one
    # strongest option per branch.
    by_branch = {}
    for prediction in pair_pool:
        by_branch.setdefault(prediction.actions[0].branch_id, []).append(prediction)
    diverse = []
    for branch in sorted(by_branch):
        options = by_branch[branch]
        efficient = min(
            options,
            key=lambda item: (
                item.buffers,
                item.area,
                -item.distribution_gain,
                item.ids,
            ),
        )
        strong = max(
            options,
            key=lambda item: (
                item.distribution_gain,
                -item.buffers,
                -item.area,
                item.ids,
            ),
        )
        diverse.append(efficient)
        if strong.ids != efficient.ids:
            diverse.append(strong)
    diverse.sort(
        key=lambda item: (
            -(item.distribution_gain / max(1, item.buffers)),
            item.buffers,
            item.area,
            item.ids,
        )
    )
    diverse = diverse[:_PAIR_ACTION_LIMIT]

    pairs = []
    for index, left in enumerate(diverse):
        for right in diverse[index + 1 :]:
            prediction = _predict(
                (left.actions[0], right.actions[0]),
                groups,
                max_buffers,
                max_area,
            )
            if (
                prediction is not None
                and baseline - prediction.worst_span >= minimum_gain
            ):
                pairs.append(prediction)

    return {
        prediction.ids: prediction for prediction in singles + pairs
    }.values()


def plan_clock_compensation(context):
    """Return exactly one timing-screened plan of native action IDs.

    Invalid, ambiguous, aligned, timing-limited, or resource-limited contexts
    intentionally produce an empty plan.  No counterfactual physical report is
    requested and no database is edited here.
    """

    empty = {"actions": []}
    if not isinstance(context, dict) or context.get("schema") != _SCHEMA:
        return empty
    groups = _groups(context)
    timing = _timing(context)
    limits = _limits(context)
    if groups is None or timing is None or limits is None:
        return empty
    paths, hold_violators, tolerances = timing
    max_buffers, max_area = limits
    if max_buffers == 0 or max_area == 0.0:
        return empty

    actions = _actions(context, groups)
    if not actions:
        return empty

    # Actions only delay sinks. If an exact earliest sink in any currently
    # worst group has no supported exclusive action, that group's span cannot
    # decrease, so neither can the global objective. This also avoids pair
    # look-ahead when a macro clock is not actionable.
    baseline_worst = max(group.span for group in groups)
    actionable_sinks = set().union(*(action.sinks for action in actions))
    for group in groups:
        if abs(group.span - baseline_worst) > 1.0e-12:
            continue
        earliest = min(group.arrivals.values())
        if any(
            arrival == earliest and sink not in actionable_sinks
            for sink, arrival in group.arrivals.items()
        ):
            return empty

    minimum_gain = max(_MINIMUM_GAIN_NS, 10.0 * max(tolerances.values()))
    candidates = list(
        _candidates(actions, groups, max_buffers, max_area, minimum_gain)
    )
    candidates.sort(
        key=lambda item: (
            item.worst_span,
            item.total_span,
            item.buffers,
            item.area,
            -item.distribution_gain,
            len(item.actions),
            item.ids,
        )
    )
    for candidate in candidates:
        if _timing_safe(candidate, paths, hold_violators, tolerances):
            return {"actions": list(candidate.ids)}
    return empty


__all__ = ["plan_clock_compensation"]

