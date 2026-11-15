import json
from types import SimpleNamespace

import pytest

from local_bench.evaluation import _replay_start_stage


def test_partition_replay_adopts_frozen_timing_at_real_step_15():
    spec = SimpleNamespace(start_stage="partition")
    config = {"partition": {"timing_input": "/frozen/features"}}
    plan = json.dumps({"plan": [{"stage": "timing_extract", "skip": None},
                                {"stage": "partition", "skip": None}]})
    assert _replay_start_stage(spec, config, plan) == "timing_extract"
    assert config == {"partition": {"timing_input": "/frozen/features"}}


@pytest.mark.parametrize("stages", [[], [{"stage": "partition", "skip": None}],
                                    [{"stage": "timing_extract", "skip": "not applicable"}]])
def test_historical_or_inapplicable_step_15_retains_partition(stages):
    assert _replay_start_stage(SimpleNamespace(start_stage="partition"),
        {"partition": {"timing_input": "/frozen/features"}}, json.dumps({"plan": stages})) == "partition"


def test_non_timing_or_other_stage_does_not_change_boundary():
    assert _replay_start_stage(SimpleNamespace(start_stage="partition"), {}, "") == "partition"
    assert _replay_start_stage(SimpleNamespace(start_stage="placement"),
        {"partition": {"timing_input": "/frozen/features"}}, "") == "placement"


@pytest.mark.parametrize("text", ["", "{}", "null", "not json"])
def test_unreadable_trusted_plan_is_not_silently_accepted(text):
    with pytest.raises(ValueError, match="trusted flow plan"):
        _replay_start_stage(SimpleNamespace(start_stage="partition"),
            {"partition": {"timing_input": "/frozen/features"}}, text)
