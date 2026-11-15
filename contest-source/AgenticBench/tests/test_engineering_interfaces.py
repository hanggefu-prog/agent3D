import copy
import pytest
from engineering_bench.interfaces import validate_clock_plan, validate_branch_map


def clock():
    return {"schema": "clock-compensation-context-v1", "actions": {
        "A": {"qualified": True, "exclusive": True, "branch_id": "top/subtree/1",
              "affected_original_sinks": ["top/reg1/CK"], "placement_site_ids": ["top/site1"],
              "added_buffers": 1, "added_area_um2": 2.},
        "B": {"qualified": True, "exclusive": True, "branch_id": "bottom/subtree/2",
              "affected_original_sinks": ["bottom/reg2/CK"], "placement_site_ids": ["bottom/site2"],
              "added_buffers": 1, "added_area_um2": 2.}},
        "limits": {"max_added_buffers": 2, "max_added_area_um2": 4.}}


def hbt():
    context = {"schema": "multi-hbt-context-v1", "target_nets": {
        "data": {"original_remote_receivers": ["top/a/D", "top/b/D"],
                 "original_hbt_id": "h1", "original_site_id": "s1", "original_driver": "bottom/g/Z"}},
        "spare_site_ids": ["s2"], "limits": {"max_added_hbts": 1}}
    record = {"logical_net": "data", "original_driver": "bottom/g/Z",
        "noninverting_buffer_proof": True, "single_driver": True}
    records = {
        "b1": dict(record, original_remote_receivers=["top/a/D"], hbt_id="h1", site_id="s1",
                   source_electrical_net="bottom/n1", sink_electrical_net="top/n1"),
        "b2": dict(record, original_remote_receivers=["top/b/D"], hbt_id="h2", site_id="s2",
                   source_electrical_net="bottom/n2", sink_electrical_net="top/n2")}
    return context, {"data": ["b1", "b2"]}, records


def test_clock_noop_and_independent_die_actions():
    context = clock()
    assert validate_clock_plan(context, {"actions": []})["added_buffers"] == 0
    assert validate_clock_plan(context, {"actions": ["A", "B"]})["added_buffers"] == 2


@pytest.mark.parametrize("mutate", [
    lambda c: c["actions"]["A"].update(exclusive=False),
    lambda c: c["actions"]["A"].update(qualified=False),
    lambda c: c["actions"]["A"].update(added_area_um2=float("nan")),
    lambda c: c["actions"]["B"].update(affected_original_sinks=["top/reg1/CK"]),
    lambda c: c["actions"]["B"].update(placement_site_ids=["top/site1"]),
    lambda c: c["limits"].update(max_added_buffers=1),
])
def test_unsafe_clock_plans_rejected(mutate):
    context = clock()
    mutate(context)
    with pytest.raises(ValueError):
        validate_clock_plan(context, {"actions": ["A", "B"]})


def test_isolated_hbt_branches_and_original_sink_coverage():
    context, mapping, records = hbt()
    assert validate_branch_map(context, mapping, records)["added_hbts"] == 1
    mapping["data"] = ["b1"]
    records.pop("b2")
    records["b1"]["original_remote_receivers"] = ["top/a/D", "top/b/D"]
    assert validate_branch_map(context, mapping, records)["added_hbts"] == 0


@pytest.mark.parametrize("mutate", [
    lambda c, m, r: r["b2"].update(source_electrical_net="bottom/n1"),
    lambda c, m, r: r["b2"].update(sink_electrical_net="top/n1"),
    lambda c, m, r: r["b2"].update(original_remote_receivers=["top/a/D"]),
    lambda c, m, r: r["b2"].update(original_driver="bottom/wrong/Z"),
    lambda c, m, r: r["b2"].update(noninverting_buffer_proof=False),
    lambda c, m, r: r["b2"].update(single_driver=False),
    lambda c, m, r: r["b1"].update(site_id="s2"),
    lambda c, m, r: c["limits"].update(max_added_hbts=0),
    lambda c, m, r: m.update(data=["b1", "b1"]),
    lambda c, m, r: r["b2"].update(hbt_id="h1"),
])
def test_bad_multibranch_materializations_rejected(mutate):
    context, mapping, records = hbt()
    mutate(context, mapping, records)
    with pytest.raises(ValueError):
        validate_branch_map(context, mapping, records)
