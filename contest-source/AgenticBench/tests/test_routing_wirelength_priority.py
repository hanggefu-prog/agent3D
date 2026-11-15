"""Wirelength-priority selection without changing native routing or hook gates."""
import json

import pytest

from local_bench.metrics import extract_metrics


HOOKS = {
    "routing_net_ordering_source": (
        "OPEN3D_NET_ORDER die={die} phase=0 nets=100 overflow_nets=20 "
        "changed_ratio=0 overflow_changed_ratio=0 top10_overlap=1 "
        "score_min=0 score_max=1 baseline_hash=11 learned_hash=11"
    ),
    "routing_ripup_policy_source": (
        "OPEN3D_RIPUP_POLICY die={die} calls=20 changed=0 score=0.5"
    ),
    "routing_maze_cost_source": (
        "OPEN3D_MAZE_COST die={die} calls=200 changed=0 multiplier_last=1"
    ),
    "routing_iteration_controller_source": (
        "OPEN3D_ITERATION_CONTROL die={die} calls=2 changed=0 cost_scale=1"
    ),
}


def write_run(root, kind, *, overflow=1000.0, wirelength=100.0, hook=True, standalone=False):
    stage = root / ("80_route" if standalone else "75_global_route")
    stage.mkdir(parents=True)
    dies = {}
    for die in ("bottom", "top"):
        dies[die] = {
            "status": "success", "grt_total_overflow": overflow,
            "grt_congestion_bins": 2, "grt_wirelength_um": wirelength,
        }
        folder = stage / die
        folder.mkdir()
        (folder / ("grt.log" if standalone else "route.log")).write_text(
            HOOKS[kind].format(die=die) + "\n" if hook else "",
            encoding="utf-8",
        )
    (stage / ("grt_metrics.json" if standalone else "metrics.json")).write_text(json.dumps({
        "stage": "global_route", "status": "success", "dies": dies,
        "ready_for_detailed_route": True, "ready_for_extract": False,
        "unexpected_warnings": 0,
    }), encoding="utf-8")
    if standalone:
        (stage / "global_route_checkpoint.json").write_text(json.dumps({
            "stage": "global_route", "status": "success", "metrics": "grt_metrics.json",
            "ready_for_detailed_route": True, "ready_for_extract": False,
        }))
    return root


@pytest.mark.parametrize("kind", HOOKS)
@pytest.mark.parametrize("overflow", [0.0, 1000.0])
def test_reference_is_one_and_weights_are_reported(tmp_path, kind, overflow):
    reference = write_run(tmp_path / "reference", kind, overflow=overflow)
    result = extract_metrics(kind, reference, reference)
    assert result["combined_score"] == pytest.approx(1.0)
    assert result["wirelength_weight"] == 0.9
    assert result["overflow_weight"] == 0.1


@pytest.mark.parametrize("kind", HOOKS)
def test_wirelength_improvement_wins_small_overflow_tradeoff(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind)
    short = write_run(tmp_path / "short", kind, overflow=1100, wirelength=90)
    low_overflow = write_run(tmp_path / "low-overflow", kind,
                             overflow=900, wirelength=110)
    short_score = extract_metrics(kind, short, reference)["combined_score"]
    overflow_score = extract_metrics(kind, low_overflow, reference)["combined_score"]
    assert short_score > 1.0 > overflow_score


@pytest.mark.parametrize("kind", HOOKS)
def test_overflow_still_has_a_selection_signal(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind)
    lower = write_run(tmp_path / "lower", kind, overflow=900)
    higher = write_run(tmp_path / "higher", kind, overflow=1100)
    assert (extract_metrics(kind, lower, reference)["combined_score"] > 1.0
            > extract_metrics(kind, higher, reference)["combined_score"])


@pytest.mark.parametrize("kind", HOOKS)
def test_shorter_wire_cannot_bypass_hook_gate(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind)
    candidate = write_run(tmp_path / "candidate", kind, wirelength=1, hook=False)
    result = extract_metrics(kind, candidate, reference)
    assert result["grt_complete"] == result["combined_score"] == 0


@pytest.mark.parametrize("kind", HOOKS)
def test_missing_wirelength_is_not_a_perfect_result(tmp_path, kind):
    reference = write_run(tmp_path / "reference", kind)
    candidate = write_run(tmp_path / "candidate", kind, wirelength=None)
    with pytest.raises(ValueError, match="wire length"):
        extract_metrics(kind, candidate, reference)


@pytest.mark.parametrize("kind", HOOKS)
@pytest.mark.parametrize("hook", [False, True])
def test_standalone_uses_grt_not_post_drt_metrics_or_logs(tmp_path, kind, hook):
    reference = write_run(tmp_path / "reference", kind, standalone=True)
    candidate = write_run(tmp_path / "candidate", kind, standalone=True, hook=hook)
    # Poison later-stage files: neither their QoR nor their hooks may be used.
    stage = candidate / "80_route"
    (stage / "metrics.json").write_text('{"stage":"route","dies":{}}')
    for die in ("bottom", "top"):
        (stage / die / "route.log").write_text(HOOKS[kind].format(die=die))
    result = extract_metrics(kind, candidate, reference)
    assert result["combined_score"] == pytest.approx(1.0 if hook else 0.0)
    checkpoint = stage / "global_route_checkpoint.json"
    checkpoint.write_text('{"stage":"route","status":"success"}')
    with pytest.raises(ValueError, match="standalone GRT checkpoint"):
        extract_metrics(kind, candidate, reference)
