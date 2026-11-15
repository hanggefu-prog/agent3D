import copy
import pytest
from engineering_bench.legalization import restoration, POLICY, ports_geometry
from engineering_bench.tasks import load_task


def row(*, master="BUF_X4", clock=0, x=10, status="PLACED", kind="CORE"):
    return {"master": master, "clock_cell": clock, "x_dbu": x, "y_dbu": 20,
            "orient": "R0", "status": status, "master_type": kind}


def test_only_data_coordinates_are_restored():
    before = {"data": row(), "old_clock": row(clock=1)}
    after = {"data": row(x=999), "new_clock": row(clock=1, x=300)}
    restore, removed = restoration(before, after)
    assert restore == "data\tBUF_X4\t10\t20\tR0\tPLACED\t0\n"
    assert removed == ["old_clock"]
    assert "new_clock" not in restore


@pytest.mark.parametrize("before,after", [
    ({"d": row()}, {}),
    ({"d": row()}, {"d": row(master="BUF_X8")}),
    ({"d": row()}, {"d": row(), "new": row(clock=0)}),
    ({"hbt": row(kind="COVER", status="FIRM")}, {"hbt": row(kind="COVER", x=900, status="PLACED")}),
    ({"macro": row(kind="BLOCK")}, {"macro": row(kind="BLOCK", x=900)}),
])
def test_data_deletion_sizing_new_logic_or_fixed_geometry_rejected(before, after):
    with pytest.raises(ValueError):
        restoration(before, after)


def test_function_and_module_tasks_keep_strict_original_placement():
    for task in ("clock_compensation", "multi_hbt_fanout"):
        assert "fixed_cell_placement_check" in load_task(task).data["required_capabilities"]
    cts = load_task("cross_die_cts")
    assert "harness_owned_data_legalization" in cts.data["required_capabilities"]
    assert "fixed_data_placement_check" not in cts.data["required_capabilities"]
    assert "restores original" in cts.instruction
    assert "not a violation" in cts.judge_instruction
    assert POLICY["candidate_editable"] is False


def test_clock_port_net_may_change_geometry_may_not(tmp_path):
    header = "port\tnet\tdirection\tsignal_type\tlayer\txmin\tymin\txmax\tymax\tstatus\n"
    before, after = tmp_path / "before.tsv", tmp_path / "after.tsv"
    before.write_text(header + "clock\told_net\tOUTPUT\tSIGNAL\tmetal10\t1\t2\t3\t4\tFIRM\n")
    after.write_text(header + "clock\tnew_leaf\tOUTPUT\tSIGNAL\tmetal10\t1\t2\t3\t4\tFIRM\n")
    assert ports_geometry(before) == ports_geometry(after)
    after.write_text(header + "clock\tnew_leaf\tOUTPUT\tSIGNAL\tmetal10\t9\t2\t3\t4\tFIRM\n")
    assert ports_geometry(before) != ports_geometry(after)
