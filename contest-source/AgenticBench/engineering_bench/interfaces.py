"""Independent structural validators for the two narrow engineering interfaces.

Context and physical records MUST come from a protected native producer. These
checks do not replace OpenDB legality, Liberty proof, RC/STA, or qualification.
They intentionally know no search algorithm and never assign a physical score.
"""
from .metrics import number


def _ids(values, label):
    if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError("Invalid " + label)
    if len(values) != len(set(values)):
        raise ValueError("Repeated " + label)
    return set(values)


def validate_clock_plan(context, plan):
    """A plan selects opaque, native-qualified physical action IDs.

    The context can contain arbitrarily rich reports, arrays and helper data.
    The small write interface avoids accepting arbitrary Tcl/report modifications
    and does not restrict Python expression syntax, helper functions or algorithms.
    """
    if context.get("schema") not in {"clock-compensation-context-v1", "clock-compensation-context-v2"}:
        raise ValueError("Unsupported clock context")
    if not isinstance(plan, dict) or set(plan) != {"actions"}:
        raise ValueError("Return an actions list")
    chosen = _ids(plan["actions"], "clock action")
    options = context["actions"]
    if not chosen <= set(options):
        raise ValueError("Unknown physical clock action")
    branches, occupied, receivers = set(), set(), set()
    count, area = 0, 0.
    for key in sorted(chosen):
        option = options[key]
        if option.get("qualified") is not True or option.get("exclusive") is not True:
            raise ValueError("Action is unqualified or affects a shared trunk")
        branch = option["branch_id"]
        if branch in branches:
            raise ValueError("Conflicting alternatives for one clock branch")
        branches.add(branch)
        sinks = _ids(option["affected_original_sinks"], "clock receiver")
        if not sinks or sinks & receivers:
            raise ValueError("Overlapping/empty affected clock branches")
        receivers.update(sinks)
        sites = _ids(option["placement_site_ids"], "buffer placement site")
        if sites & occupied:
            raise ValueError("Clock actions compete for the same legal space")
        occupied.update(sites)
        buffers = option["added_buffers"]
        if type(buffers) is not int or buffers < 1:
            raise ValueError("A physical action must add a positive buffer count")
        size = number(option["added_area_um2"], "clock buffer area")
        if size <= 0:
            raise ValueError("Invalid buffer area")
        count += buffers
        area += size
    limits = context["limits"]
    if type(limits["max_added_buffers"]) is not int or limits["max_added_buffers"] < 0:
        raise ValueError("Invalid clock buffer budget")
    if count > limits["max_added_buffers"] or area > number(limits["max_added_area_um2"], "clock area budget"):
        raise ValueError("Clock resource budget exceeded")
    return {"actions": sorted(chosen), "added_buffers": count, "added_area_um2": area,
            "requires_native_validation": True}


def validate_branch_map(context, mapping, physical_branches):
    """Check logical coverage against independently read-back electrical branches.

    physical_branches is a protected OpenDB/Liberty read-back, not the submitted
    map. Source and sink electrical net IDs are die-qualified native identities.
    A preserved one-HBT net is also recorded; omissions cannot improve an average.
    """
    if context.get("schema") != "multi-hbt-context-v1":
        raise ValueError("Unsupported multi-HBT context")
    targets = context["target_nets"]
    if not isinstance(mapping, dict) or set(mapping) != set(targets):
        raise ValueError("Every fixed target must occur in the branch map")
    seen_branches, seen_hbts, electrical = set(), set(), set()
    total_added = 0
    for net, target in targets.items():
        branches = _ids(mapping[net], "physical branch")
        if not 1 <= len(branches) <= 3:
            raise ValueError("Use one to three branches per target net")
        if branches & seen_branches:
            raise ValueError("Physical branch assigned to two logical nets")
        seen_branches.update(branches)
        original = _ids(target["original_remote_receivers"], "original remote receiver")
        if not original:
            raise ValueError("Empty target receiver inventory")
        original_hbt = target["original_hbt_id"]
        original_preserved, covered = False, set()
        for branch in sorted(branches):
            if branch not in physical_branches:
                raise ValueError("Branch missing from native read-back")
            row = physical_branches[branch]
            if row["logical_net"] != net or row["original_driver"] != target["original_driver"]:
                raise ValueError("Branch carries the wrong original logical driver")
            if row.get("noninverting_buffer_proof") is not True or row.get("single_driver") is not True:
                raise ValueError("Branch lacks native logic/isolation proof")
            sinks = _ids(row["original_remote_receivers"], "branch receiver")
            if not sinks or sinks & covered or not sinks <= original:
                raise ValueError("Branch receivers overlap, disappear or change")
            covered.update(sinks)
            hbt = row["hbt_id"]
            if hbt in seen_hbts:
                raise ValueError("HBT assigned more than once")
            seen_hbts.add(hbt)
            if hbt == original_hbt:
                original_preserved = True
                if row["site_id"] != target["original_site_id"]:
                    raise ValueError("Original HBT moved")
            else:
                if row["site_id"] not in context["spare_site_ids"]:
                    raise ValueError("New HBT is not on a reserved legal site")
                total_added += 1
            for field in ("source_electrical_net", "sink_electrical_net"):
                identity = row[field]
                if not isinstance(identity, str) or not identity or identity in electrical:
                    raise ValueError("Multiple HBTs share an electrical net; pairwise RC is invalid")
                electrical.add(identity)
        if covered != original or not original_preserved:
            raise ValueError("Original receivers or existing HBT not preserved")
    if set(physical_branches) != seen_branches:
        raise ValueError("Unmapped physical branches")
    site_ids = [row["site_id"] for row in physical_branches.values()]
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("Overlapping HBT sites")
    budget = context["limits"]["max_added_hbts"]
    if type(budget) is not int or budget < 0 or total_added > budget:
        raise ValueError("HBT resource budget exceeded")
    return {"target_count": len(targets), "branch_count": len(seen_branches),
            "added_hbts": total_added, "requires_native_validation": True}
